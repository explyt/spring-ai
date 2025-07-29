/*
 * Copyright 2023 - 2024 the original author or authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.springframework.ai.google.gemini;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.JsonSerializer;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.ai.chat.messages.AssistantMessage;
import org.springframework.ai.chat.messages.MessageType;
import org.springframework.ai.chat.messages.ToolResponseMessage;
import org.springframework.ai.chat.messages.UserMessage;
import org.springframework.ai.chat.metadata.ChatResponseMetadata;
import org.springframework.ai.chat.metadata.EmptyUsage;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.chat.model.Generation;
import org.springframework.ai.chat.model.StreamingChatModel;
import org.springframework.ai.chat.prompt.ChatOptions;
import org.springframework.ai.chat.prompt.Prompt;
import org.springframework.ai.google.gemini.api.GoogleGeminiApi;
import org.springframework.ai.google.gemini.api.GoogleGeminiApi.ChatCompletion;
import org.springframework.ai.google.gemini.api.GoogleGeminiApi.ChatCompletionRequest;
import org.springframework.ai.model.tool.ToolCallingManager;
import org.springframework.ai.model.tool.ToolExecutionEligibilityPredicate;
import org.springframework.ai.model.tool.DefaultToolExecutionEligibilityPredicate;
import org.springframework.ai.model.tool.ToolExecutionResult;
import org.springframework.ai.model.tool.ToolCallingChatOptions;
import org.springframework.ai.tool.definition.ToolDefinition;
import org.springframework.ai.google.gemini.metadata.GoogleGeminiUsage;
import org.springframework.ai.google.gemini.api.schema.GeminiToolCallingManager;
import org.springframework.ai.retry.RetryUtils;
import org.springframework.http.ResponseEntity;
import org.springframework.retry.support.RetryTemplate;
import org.springframework.util.Assert;
import reactor.core.publisher.Flux;

import java.util.HashMap;

import java.util.*;
import java.util.stream.Stream;

/**
 * @author Geng Rong
 */

public class GoogleGeminiChatModel implements ChatModel, StreamingChatModel {

	private static final Logger logger = LoggerFactory.getLogger(GoogleGeminiChatModel.class);

	/**
	 * The default options used for the chat completion requests.
	 */
	private final GoogleGeminiChatOptions defaultOptions;

	/**
	 * The retry template used to retry the Google Gemini API calls.
	 */
	public final RetryTemplate retryTemplate;

	/**
	 * Low-level access to the Google Gemini API.
	 */
	private final GoogleGeminiApi api;

	/**
	 * Tool calling manager for function/tool call support.
	 */
	private final GeminiToolCallingManager toolCallingManager;

	/**
	 * Predicate to determine if tool execution is required.
	 */
	private final ToolExecutionEligibilityPredicate toolExecutionEligibilityPredicate;

	/**
	 * Creates an instance of the GoogleGeminiChatModel.
	 *
	 * @param api The GoogleGeminiApi instance to be used for interacting with the Google
	 *            Gemini Chat API.
	 * @throws IllegalArgumentException if api is null
	 */
	public GoogleGeminiChatModel(GoogleGeminiApi api) {
		this(api, GoogleGeminiChatOptions.builder().withTemperature(1D).build());
	}

	public GoogleGeminiChatModel(GoogleGeminiApi api, GoogleGeminiChatOptions options) {
		this(api, options, RetryUtils.DEFAULT_RETRY_TEMPLATE);
	}

	public GoogleGeminiChatModel(GoogleGeminiApi api, GoogleGeminiChatOptions options, RetryTemplate retryTemplate) {
		this(api, options, new GeminiToolCallingManager(ToolCallingManager.builder().build()), retryTemplate, new DefaultToolExecutionEligibilityPredicate());
	}

	public GoogleGeminiChatModel(GoogleGeminiApi api, GoogleGeminiChatOptions options, GeminiToolCallingManager toolCallingManager, RetryTemplate retryTemplate, ToolExecutionEligibilityPredicate toolExecutionEligibilityPredicate) {
		Assert.notNull(api, "GoogleGeminiApi must not be null");
		Assert.notNull(options, "Options must not be null");
		Assert.notNull(toolCallingManager, "ToolCallingManager must not be null");
		Assert.notNull(retryTemplate, "RetryTemplate must not be null");
		Assert.notNull(toolExecutionEligibilityPredicate, "ToolExecutionEligibilityPredicate must not be null");
		this.api = api;
		this.defaultOptions = options;
		this.toolCallingManager = toolCallingManager;
		this.retryTemplate = retryTemplate;
		this.toolExecutionEligibilityPredicate = toolExecutionEligibilityPredicate;
	}

	private ObjectMapper jacksonObjectMapper = new ObjectMapper();

	private AssistantMessage createAssistantMessageFromCandidate(GoogleGeminiApi.Candidate choice) {
		String message = null;
		List<AssistantMessage.ToolCall> functionCalls = Collections.emptyList();
		if (choice != null && choice.content() != null && choice.content().parts() != null
				&& !choice.content().parts().isEmpty()) {
			message = choice.content().parts().get(0).text();

			functionCalls = choice
					.content()
					.parts()
					.stream()
					.map(GoogleGeminiApi.Part::functionCall)
					.filter(Objects::nonNull)
					.map(functionCall -> {
						try {
							return new AssistantMessage.ToolCall(
									functionCall.id(),
									"function_call",
									functionCall.name(),
									jacksonObjectMapper.writeValueAsString(functionCall.args())
							);
						} catch (JsonProcessingException e) {
							throw new RuntimeException(e);
						}
					}).toList();
		}

		return new AssistantMessage(message, Collections.emptyMap(), functionCalls);
	}

	@Override
	public ChatResponse call(Prompt prompt) {
		return internalCall(prompt, null);
	}

	private ChatResponse internalCall(Prompt prompt, ChatResponse previousChatResponse) {
		ChatCompletionRequest request = createRequest(prompt);

		ChatResponse response = this.retryTemplate.execute(ctx -> {
			ResponseEntity<ChatCompletion> completionEntity = this.doChatCompletion(request);
			var chatCompletion = completionEntity.getBody();
			if (chatCompletion == null) {
				logger.warn("No chat completion returned for prompt: {}", prompt);
				return new ChatResponse(List.of());
			}
			List<Generation> generations = chatCompletion.choices()
					.stream()
					.map(choice -> new Generation(createAssistantMessageFromCandidate(choice)))
					.toList();
			return new ChatResponse(generations, from(completionEntity.getBody()));
		});

		if (this.toolExecutionEligibilityPredicate.isToolExecutionRequired(prompt.getOptions(), response)) {
			var toolExecutionResult = this.toolCallingManager.executeToolCalls(prompt, response);
			if (toolExecutionResult.returnDirect()) {
				// Return tool execution result directly to the client.
				return ChatResponse.builder()
						.from(response)
						.generations(ToolExecutionResult.buildGenerations(toolExecutionResult))
						.build();
			} else {
				// Send the tool execution result back to the model.
				return this.internalCall(new Prompt(toolExecutionResult.conversationHistory(), prompt.getOptions()), response);
			}
		}
		return response;
	}

	private ChatResponseMetadata from(GoogleGeminiApi.ChatCompletion result) {
		Assert.notNull(result, "Google Gemini ChatCompletionResult must not be null");
		return ChatResponseMetadata.builder()
				.usage(result.usage() == null ? new EmptyUsage() : GoogleGeminiUsage.from(result.usage()))
				.build();
	}

	@Override
	public ChatOptions getDefaultOptions() {
		return GoogleGeminiChatOptions.fromOptions(this.defaultOptions);
	}

	@Override
	public Flux<ChatResponse> stream(Prompt prompt) {
		return internalStream(prompt, null);
	}

	private Flux<ChatResponse> internalStream(Prompt prompt, ChatResponse previousChatResponse) {
		ChatCompletionRequest request = createRequest(prompt);
		return retryTemplate.execute(ctx -> {
			var completionChunks = this.api.chatCompletionStream(request);
			return completionChunks.concatMap(chatCompletion -> {
				List<Generation> generations = chatCompletion.choices()
						.stream()
						.map(choice -> new Generation(createAssistantMessageFromCandidate(choice)))
						.toList();
				ChatResponse response = new ChatResponse(generations, from(chatCompletion));
				if (this.toolExecutionEligibilityPredicate.isToolExecutionRequired(prompt.getOptions(), response)) {
					var toolExecutionResult = this.toolCallingManager.executeToolCalls(prompt, response);
					if (toolExecutionResult.returnDirect()) {
						return Flux.just(ChatResponse.builder()
								.from(response)
								.generations(ToolExecutionResult.buildGenerations(toolExecutionResult))
								.build());
					} else {
						return this.internalStream(new Prompt(toolExecutionResult.conversationHistory(), prompt.getOptions()), response);
					}
				}
				return Flux.just(response);
			});
		});
	}

	protected ResponseEntity<ChatCompletion> doChatCompletion(ChatCompletionRequest request) {
		return this.api.chatCompletionEntity(request);
	}

	/**
	 * Accessible for testing.
	 */
	ChatCompletionRequest createRequest(Prompt prompt) {
		GoogleGeminiChatOptions options = null;
		if (prompt.getOptions() != null) {
			if (prompt.getOptions() instanceof GoogleGeminiChatOptions googleGeminiChatOptions) {
				options = googleGeminiChatOptions;
			}

		}
		// TODO: WHY MERGILKA EATS MY TOOLS?
//        if (this.defaultOptions != null) {
//            options = ModelOptionsUtils.merge(options, this.defaultOptions, GoogleGeminiChatOptions.class);
//        }

		// Add tool definitions if present
		List<ToolDefinition> toolDefinitions = this.toolCallingManager != null /* && options instanceof ToolCallingChatOptions toolOptions */
				? this.toolCallingManager.resolveToolDefinitions(options)
				: List.of();

		ChatCompletionRequest request;
		if (!toolDefinitions.isEmpty()) {
			// Convert ToolDefinition to Gemini Tool format
			List<GoogleGeminiApi.FunctionDeclaration> functionDeclarations = toolDefinitions.stream()
				.map(td -> new GoogleGeminiApi.FunctionDeclaration(
					td.name(),
					td.description(),
					org.springframework.ai.model.ModelOptionsUtils.jsonToMap(td.inputSchema())
				))
				.toList();

			List<GoogleGeminiApi.ChatCompletionMessage> chatCompletionMessages = prompt.getInstructions().stream()
				.filter(i -> i.getMessageType() != MessageType.SYSTEM)
				.map(msg -> {
					if (msg instanceof AssistantMessage assistantMessage) {
						Collection<GoogleGeminiApi.Part.FunctionCall> toolCalls = assistantMessage.hasToolCalls()
							? assistantMessage
							.getToolCalls()
							.stream()
							.map(call -> {
								try {
									return new GoogleGeminiApi.Part.FunctionCall(
										call.id(),
										call.name(),
											jacksonObjectMapper.readValue(call.arguments(), new TypeReference<HashMap<String, String>>() {}
										)
									);
								} catch (JsonProcessingException e) {
									throw new RuntimeException(e);
								}
							}).toList()
							: Collections.emptyList();

						List<GoogleGeminiApi.Part> parts = new ArrayList<>();

						for (GoogleGeminiApi.Part.FunctionCall call : toolCalls) {
							parts.add(new GoogleGeminiApi.Part(call));
						}

						GoogleGeminiApi.ChatCompletionMessage message = new GoogleGeminiApi.ChatCompletionMessage(GoogleGeminiApi.ChatCompletionMessage.Role.ASSISTANT, parts);

						return message;
					} else if (msg instanceof UserMessage userMessage) {
						GoogleGeminiApi.ChatCompletionMessage message = new GoogleGeminiApi.ChatCompletionMessage(
							GoogleGeminiApi.ChatCompletionMessage.Role.USER, userMessage.getText()
						);

						return message;
					} else if (msg instanceof ToolResponseMessage toolResponseMessage) {
						Collection<GoogleGeminiApi.Part.FunctionResponse> functionResponses = toolResponseMessage.getResponses()
							.stream()
							.map(functionResponse ->
								{
									try {
										return new GoogleGeminiApi.Part.FunctionResponse(
											functionResponse.id(),
											functionResponse.name(),
											jacksonObjectMapper.readValue(functionResponse.responseData(), new TypeReference<HashMap<String, String>>() {})
										);
									} catch (JsonProcessingException e) {
										throw new RuntimeException(e);
									}
									}
							)
								.toList();

						List<GoogleGeminiApi.Part> parts = new ArrayList<>();

						for (GoogleGeminiApi.Part.FunctionResponse functionResponse : functionResponses) {
							parts.add(new GoogleGeminiApi.Part(functionResponse));
						}

						GoogleGeminiApi.ChatCompletionMessage message = new GoogleGeminiApi.ChatCompletionMessage(GoogleGeminiApi.ChatCompletionMessage.Role.TOOL, parts);

						return message;
					} else {
						throw new RuntimeException("Unknown instance of message");
					}
				}).toList();

			GoogleGeminiApi.Tool tool = new GoogleGeminiApi.Tool(functionDeclarations);
			request = new ChatCompletionRequest(
					chatCompletionMessages,
					GoogleGeminiApi.ChatCompletionMessage.getSystemInstruction(prompt),
					GoogleGeminiApi.GenerationConfig.of(options),
					List.of(tool)
			);
		} else {
			request = new ChatCompletionRequest(prompt, options);
		}
		return request;
	}

	public static Builder builder() {
		return new Builder();
	}

	public static class Builder {
		private String apiKey;
		private GoogleGeminiChatOptions options = GoogleGeminiChatOptions.builder().build();
		private RetryTemplate retryTemplate = RetryUtils.DEFAULT_RETRY_TEMPLATE;
		private GeminiToolCallingManager toolCallingManager = new GeminiToolCallingManager(ToolCallingManager.builder().build());
		private ToolExecutionEligibilityPredicate toolExecutionEligibilityPredicate = new DefaultToolExecutionEligibilityPredicate();

		public Builder apiKey(String apiKey) {
			this.apiKey = apiKey;
			return this;
		}

		public Builder options(GoogleGeminiChatOptions options) {
			this.options = options;
			return this;
		}

		public Builder toolCallingManager(GeminiToolCallingManager toolCallingManager) {
			this.toolCallingManager = toolCallingManager;
			return this;
		}

		public Builder toolExecutionEligibilityPredicate(ToolExecutionEligibilityPredicate predicate) {
			this.toolExecutionEligibilityPredicate = predicate;
			return this;
		}

		public Builder retryTemplate(RetryTemplate retryTemplate) {
			this.retryTemplate = retryTemplate;
			return this;
		}

		public GoogleGeminiChatModel build() {
			Assert.hasText(apiKey, "API key must not be empty");
			return new GoogleGeminiChatModel(new GoogleGeminiApi(apiKey), options, toolCallingManager, retryTemplate, toolExecutionEligibilityPredicate);
		}
	}
}


