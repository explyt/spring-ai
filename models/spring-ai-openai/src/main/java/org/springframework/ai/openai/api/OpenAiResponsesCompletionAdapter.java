/*
 * Copyright 2023-2025 the original author or authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.springframework.ai.openai.api;

import java.util.ArrayList;
import java.util.List;

import tools.jackson.databind.JsonNode;

import org.springframework.ai.model.ModelOptionsUtils;
import org.springframework.ai.openai.api.OpenAiApi.ChatCompletion;
import org.springframework.ai.openai.api.OpenAiApi.ChatCompletion.Choice;
import org.springframework.ai.openai.api.OpenAiApi.ChatCompletionMessage;
import org.springframework.ai.openai.api.OpenAiApi.ChatCompletionMessage.ChatCompletionFunction;
import org.springframework.ai.openai.api.OpenAiApi.ChatCompletionMessage.Role;
import org.springframework.ai.openai.api.OpenAiApi.ChatCompletionMessage.ToolCall;

/**
 * Adapts a non-streamed OpenAI <b>Responses API</b> {@code response} object into the Chat
 * Completions {@link ChatCompletion} shape: {@code output_text} parts become the message
 * content, {@code refusal} parts the refusal, reasoning summaries (or the raw reasoning
 * text when no summary was produced) the reasoning content, and {@code function_call}
 * items the tool calls. Finish reason and usage are translated as in
 * {@link OpenAiResponsesStreamAdapter}; a {@code failed} response is raised as
 * {@link org.springframework.ai.retry.NonTransientAiException}.
 */
public final class OpenAiResponsesCompletionAdapter {

	private static final String REASONING_SEPARATOR = "\n\n";

	private OpenAiResponsesCompletionAdapter() {
	}

	/**
	 * Converts a Responses API {@code response} object into a {@link ChatCompletion}.
	 * @param responseJson the JSON body returned by {@code POST /v1/responses}.
	 * @return the equivalent {@link ChatCompletion}.
	 */
	public static ChatCompletion toChatCompletion(String responseJson) {
		JsonNode response = ModelOptionsUtils.JSON_MAPPER.readTree(responseJson);
		if ("failed".equals(response.path("status").asText(null))) {
			JsonNode error = response.path("error");
			throw OpenAiResponsesStreamAdapter.failure(error.path("code").asText(null),
					error.path("message").asText(null));
		}
		StringBuilder text = new StringBuilder();
		StringBuilder refusal = new StringBuilder();
		List<String> reasoning = new ArrayList<>();
		List<ToolCall> toolCalls = new ArrayList<>();
		for (JsonNode item : response.path("output")) {
			switch (item.path("type").asText("")) {
				case "message" -> appendMessageParts(item, text, refusal);
				case "reasoning" -> addIfNotEmpty(reasoning, reasoningText(item));
				case "function_call" -> toolCalls.add(toolCall(item));
				default -> {
				}
			}
		}
		ChatCompletionMessage message = new ChatCompletionMessage(emptyToNull(text), Role.ASSISTANT, null, null,
				toolCalls.isEmpty() ? null : toolCalls, emptyToNull(refusal), null, null,
				reasoning.isEmpty() ? null : String.join(REASONING_SEPARATOR, reasoning), null);
		Choice choice = new Choice(OpenAiResponsesStreamAdapter.finishReason(response, !toolCalls.isEmpty()), 0,
				message, null);
		JsonNode createdAt = response.path("created_at");
		return new ChatCompletion(response.path("id").asText(null), List.of(choice),
				createdAt.isNumber() ? createdAt.asLong() : null, response.path("model").asText(null), null, null,
				"chat.completion", OpenAiResponsesStreamAdapter.toUsage(response));
	}

	private static void appendMessageParts(JsonNode item, StringBuilder text, StringBuilder refusal) {
		for (JsonNode part : item.path("content")) {
			switch (part.path("type").asText("")) {
				case "output_text" -> text.append(part.path("text").asText(""));
				case "refusal" -> refusal.append(part.path("refusal").asText(""));
				default -> {
				}
			}
		}
	}

	private static String reasoningText(JsonNode item) {
		JsonNode parts = item.path("summary").isEmpty() ? item.path("content") : item.path("summary");
		List<String> texts = new ArrayList<>();
		parts.forEach(part -> addIfNotEmpty(texts, part.path("text").asText("")));
		return String.join(REASONING_SEPARATOR, texts);
	}

	private static ToolCall toolCall(JsonNode item) {
		return new ToolCall(OpenAiResponsesStreamAdapter.callId(item), "function",
				new ChatCompletionFunction(item.path("name").asText(null), item.path("arguments").asText("")));
	}

	private static void addIfNotEmpty(List<String> target, String value) {
		if (!value.isEmpty()) {
			target.add(value);
		}
	}

	private static String emptyToNull(StringBuilder value) {
		return value.isEmpty() ? null : value.toString();
	}

}
