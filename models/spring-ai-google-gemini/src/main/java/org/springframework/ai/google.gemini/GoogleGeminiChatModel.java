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

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.ai.chat.messages.AssistantMessage;
import org.springframework.ai.chat.metadata.ChatGenerationMetadata;
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
import org.springframework.ai.google.gemini.metadata.GoogleGeminiUsage;
import org.springframework.ai.model.ModelOptionsUtils;
import org.springframework.ai.retry.RetryUtils;
import org.springframework.http.ResponseEntity;
import org.springframework.retry.support.RetryTemplate;
import org.springframework.util.Assert;
import reactor.core.publisher.Flux;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.ConcurrentHashMap;

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
	 * Creates an instance of the GoogleGeminiChatModel.
	 * @param api The GoogleGeminiApi instance to be used for interacting with the Google
	 * Gemini Chat API.
	 * @throws IllegalArgumentException if api is null
	 */
	public GoogleGeminiChatModel(GoogleGeminiApi api) {
		this(api, GoogleGeminiChatOptions.builder().withTemperature(1D).build());
	}

	/**
	 * Initializes an instance of the GoogleGeminiChatModel.
	 * @param api The GoogleGeminiApi instance to be used for interacting with the Google
	 * Gemini Chat API.
	 * @param options The GoogleGeminiChatOptions to configure the chat client.
	 */
	public GoogleGeminiChatModel(GoogleGeminiApi api, GoogleGeminiChatOptions options) {
		this(api, options, RetryUtils.DEFAULT_RETRY_TEMPLATE);
	}

	/**
	 * Initializes a new instance of the GoogleGeminiChatModel.
	 * @param api The GoogleGeminiApi instance to be used for interacting with the Google
	 * Gemini Chat API.
	 * @param options The GoogleGeminiChatOptions to configure the chat client.
	 * @param retryTemplate The retry template.
	 */
	public GoogleGeminiChatModel(GoogleGeminiApi api, GoogleGeminiChatOptions options, RetryTemplate retryTemplate) {
		Assert.notNull(api, "GoogleGeminiApi must not be null");
		Assert.notNull(options, "Options must not be null");
		Assert.notNull(retryTemplate, "RetryTemplate must not be null");
		this.api = api;
		this.defaultOptions = options;
		this.retryTemplate = retryTemplate;
	}

	@Override
	public ChatResponse call(Prompt prompt) {

		ChatCompletionRequest request = createRequest(prompt);

		return this.retryTemplate.execute(ctx -> {

			ResponseEntity<ChatCompletion> completionEntity = this.doChatCompletion(request);

			var chatCompletion = completionEntity.getBody();
			if (chatCompletion == null) {
				logger.warn("No chat completion returned for prompt: {}", prompt);
				return new ChatResponse(List.of());
			}

			List<Generation> generations = chatCompletion.choices()
				.stream()
				.map(choice -> new Generation(new AssistantMessage(choice.content().parts().get(0).text()), null))
				.toList();

			return new ChatResponse(generations, from(completionEntity.getBody()));
		});
	}

	private ChatResponseMetadata from(GoogleGeminiApi.ChatCompletion result) {
		Assert.notNull(result, "Google Gemini ChatCompletionResult must not be null");
		return ChatResponseMetadata.builder()
			.withUsage(result.usage() == null ? new EmptyUsage() : GoogleGeminiUsage.from(result.usage()))
			.build();
	}

	@Override
	public ChatOptions getDefaultOptions() {
		return GoogleGeminiChatOptions.fromOptions(this.defaultOptions);
	}

	@Override
	public Flux<ChatResponse> stream(Prompt prompt) {

		ChatCompletionRequest request = createRequest(prompt);
		return retryTemplate.execute(ctx -> {
			var completionChunks = this.api.chatCompletionStream(request);

			return completionChunks.map(chatCompletion -> {
				List<Generation> generations = chatCompletion.choices().stream().map(choice -> {
					var parts = choice.content().parts();
					if (parts == null || parts.isEmpty()) {
						return null;
					}
					var generation = new Generation(new AssistantMessage(parts.get(0).text()));
					return generation;
				}).filter(Objects::nonNull).toList();
				return new ChatResponse(generations, from(chatCompletion));
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
		ChatCompletionRequest request = new ChatCompletionRequest(prompt);

		if (prompt.getOptions() != null) {
			if (prompt.getOptions() instanceof ChatOptions) {
				ChatOptions runtimeOptions = (ChatOptions) prompt.getOptions();
				GoogleGeminiChatOptions updatedRuntimeOptions = ModelOptionsUtils.copyToTarget(runtimeOptions,
						ChatOptions.class, GoogleGeminiChatOptions.class);

				request = ModelOptionsUtils.merge(updatedRuntimeOptions, request, ChatCompletionRequest.class);
			}
			else {
				throw new IllegalArgumentException("Prompt options are not of type ChatOptions: "
						+ prompt.getOptions().getClass().getSimpleName());
			}
		}

		if (this.defaultOptions != null) {
			request = ModelOptionsUtils.merge(request, this.defaultOptions, ChatCompletionRequest.class);
		}
		return request;
	}

}