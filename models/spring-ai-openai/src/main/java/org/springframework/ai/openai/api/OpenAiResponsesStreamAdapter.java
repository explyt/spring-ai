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

import tools.jackson.core.JacksonException;
import tools.jackson.databind.JsonNode;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import reactor.core.publisher.Flux;

import org.springframework.ai.model.ModelOptionsUtils;
import org.springframework.ai.openai.api.OpenAiApi.ChatCompletionChunk;
import org.springframework.ai.openai.api.OpenAiApi.ChatCompletionChunk.ChunkChoice;
import org.springframework.ai.openai.api.OpenAiApi.ChatCompletionFinishReason;
import org.springframework.ai.openai.api.OpenAiApi.ChatCompletionMessage;
import org.springframework.ai.openai.api.OpenAiApi.ChatCompletionMessage.ChatCompletionFunction;
import org.springframework.ai.openai.api.OpenAiApi.ChatCompletionMessage.Role;
import org.springframework.ai.openai.api.OpenAiApi.ChatCompletionMessage.ToolCall;
import org.springframework.ai.openai.api.OpenAiApi.Usage;
import org.springframework.util.StringUtils;

/**
 * Adapts an OpenAI <b>Responses API</b> SSE stream ({@code response.created},
 * {@code response.output_text.delta}, {@code response.output_item.done},
 * {@code response.completed}, ...) into the Chat Completions {@link ChatCompletionChunk}
 * shape, so that the typed branch of the Responses raw passthrough runs the exact same
 * chunk-to-{@code ChatResponse} pipeline (usage accumulation, pricing, audit) as the Chat
 * Completions dialect. Only the subset needed downstream is mapped:
 * <ul>
 * <li>{@code response.output_text.delta} → assistant {@code content} delta;</li>
 * <li>{@code response.refusal.delta} → {@code refusal} delta;</li>
 * <li>{@code response.reasoning_summary_text.delta} /
 * {@code response.reasoning_text.delta} → {@code reasoning_content} delta;</li>
 * <li>{@code response.output_item.done} of a {@code function_call} item → one complete
 * {@code tool_calls} entry (name + full arguments; argument deltas are not
 * replayed);</li>
 * <li>{@code response.completed} / {@code response.incomplete} → a finish-reason chunk
 * followed by a usage-only chunk (the shape OpenAI emits with
 * {@code stream_options.include_usage=true}), with Responses usage names translated to
 * Chat Completions ones.</li>
 * </ul>
 * {@code response.failed} and {@code error} events are logged and produce no typed chunk:
 * the raw branch forwards the provider frame to the client verbatim, mirroring the
 * Anthropic passthrough. Unknown event types are ignored. The adapter is stateful per
 * stream: obtain one through {@link #toChatCompletionChunks(Flux)}.
 */
public final class OpenAiResponsesStreamAdapter {

	private static final Logger logger = LoggerFactory.getLogger(OpenAiResponsesStreamAdapter.class);

	private static final String CHUNK_OBJECT = "chat.completion.chunk";

	private static final String SSE_DONE = "[DONE]";

	private String responseId = "NO_ID";

	private String model;

	private Long createdAt;

	private boolean roleSent;

	private int nextToolCallIndex;

	private boolean sawFunctionCall;

	private OpenAiResponsesStreamAdapter() {
	}

	/**
	 * Converts Responses API SSE {@code data:} payloads into Chat Completions chunks.
	 * @param sseData the raw SSE {@code data:} payloads of one Responses stream.
	 * @return the equivalent {@link ChatCompletionChunk} stream.
	 */
	public static Flux<ChatCompletionChunk> toChatCompletionChunks(Flux<String> sseData) {
		return Flux.defer(() -> {
			OpenAiResponsesStreamAdapter adapter = new OpenAiResponsesStreamAdapter();
			return sseData.concatMapIterable(adapter::onData);
		});
	}

	List<ChatCompletionChunk> onData(String data) {
		if (!StringUtils.hasText(data) || SSE_DONE.equals(data.trim())) {
			return List.of();
		}
		JsonNode event;
		try {
			event = ModelOptionsUtils.JSON_MAPPER.readTree(data);
		}
		catch (JacksonException e) {
			logger.warn("Skipping unparseable Responses SSE frame: {}", e.getMessage());
			return List.of();
		}
		String type = event.path("type").asText(null);
		if (type == null) {
			return List.of();
		}
		switch (type) {
			case "response.created", "response.in_progress", "response.queued" -> {
				captureResponseHeader(event.path("response"));
				return List.of();
			}
			case "response.output_text.delta" -> {
				return deltaChunk(event.path("delta").asText(null), null, null);
			}
			case "response.refusal.delta" -> {
				return deltaChunk(null, event.path("delta").asText(null), null);
			}
			case "response.reasoning_summary_text.delta", "response.reasoning_text.delta" -> {
				return deltaChunk(null, null, event.path("delta").asText(null));
			}
			case "response.output_item.done" -> {
				return outputItemDone(event.path("item"));
			}
			case "response.completed", "response.incomplete" -> {
				return terminalChunks(event.path("response"));
			}
			case "response.failed" -> {
				JsonNode error = event.path("response").path("error");
				logger.warn("Responses stream failed: code={} message={}", error.path("code").asText(null),
						error.path("message").asText(null));
				return List.of();
			}
			case "error" -> {
				logger.warn("Responses stream error event: code={} message={}", event.path("code").asText(null),
						event.path("message").asText(null));
				return List.of();
			}
			default -> {
				return List.of();
			}
		}
	}

	private void captureResponseHeader(JsonNode response) {
		String id = response.path("id").asText(null);
		if (StringUtils.hasText(id)) {
			this.responseId = id;
		}
		String responseModel = response.path("model").asText(null);
		if (StringUtils.hasText(responseModel)) {
			this.model = responseModel;
		}
		JsonNode created = response.path("created_at");
		if (created.isNumber()) {
			this.createdAt = created.asLong();
		}
	}

	private List<ChatCompletionChunk> deltaChunk(String content, String refusal, String reasoning) {
		if (content == null && refusal == null && reasoning == null) {
			return List.of();
		}
		ChatCompletionMessage delta = new ChatCompletionMessage(content, roleForDelta(), null, null, null, refusal,
				null, null, reasoning, null);
		return List.of(chunk(List.of(new ChunkChoice(null, 0, delta, null)), null));
	}

	private List<ChatCompletionChunk> outputItemDone(JsonNode item) {
		if (!"function_call".equals(item.path("type").asText(null))) {
			return List.of();
		}
		this.sawFunctionCall = true;
		String callId = item.path("call_id").asText(null);
		if (!StringUtils.hasText(callId)) {
			callId = item.path("id").asText(null);
		}
		String arguments = item.path("arguments").asText("");
		ToolCall toolCall = new ToolCall(this.nextToolCallIndex++, callId, "function",
				new ChatCompletionFunction(item.path("name").asText(null), arguments));
		ChatCompletionMessage delta = new ChatCompletionMessage(null, roleForDelta(), null, null, List.of(toolCall),
				null, null, null, null, null);
		return List.of(chunk(List.of(new ChunkChoice(null, 0, delta, null)), null));
	}

	private List<ChatCompletionChunk> terminalChunks(JsonNode response) {
		captureResponseHeader(response);
		List<ChatCompletionChunk> chunks = new ArrayList<>(2);
		ChatCompletionMessage emptyDelta = new ChatCompletionMessage(null, roleForDelta(), null, null, null, null, null,
				null, null, null);
		chunks.add(chunk(List.of(new ChunkChoice(finishReason(response), 0, emptyDelta, null)), null));
		Usage usage = toUsage(response);
		if (usage != null) {
			chunks.add(chunk(List.of(), usage));
		}
		return chunks;
	}

	private ChatCompletionFinishReason finishReason(JsonNode response) {
		if ("incomplete".equals(response.path("status").asText(null))) {
			String reason = response.path("incomplete_details").path("reason").asText("");
			return switch (reason) {
				case "max_output_tokens" -> ChatCompletionFinishReason.LENGTH;
				case "content_filter" -> ChatCompletionFinishReason.CONTENT_FILTER;
				default -> ChatCompletionFinishReason.STOP;
			};
		}
		return this.sawFunctionCall ? ChatCompletionFinishReason.TOOL_CALLS : ChatCompletionFinishReason.STOP;
	}

	/**
	 * Translates Responses usage ({@code input_tokens}, {@code output_tokens},
	 * {@code input_tokens_details.cached_tokens},
	 * {@code output_tokens_details.reasoning_tokens}) into the Chat Completions
	 * {@link Usage}. Gateway extensions ({@code compute_millis}, {@code wait_millis},
	 * {@code price}, {@code price_with_discount}) are taken from the usage object, or
	 * from the response root when misplaced there.
	 */
	private static Usage toUsage(JsonNode response) {
		JsonNode usage = response.path("usage");
		if (!usage.isObject()) {
			return null;
		}
		Integer promptTokens = intOrNull(usage.path("input_tokens"));
		Integer completionTokens = intOrNull(usage.path("output_tokens"));
		Integer totalTokens = intOrNull(usage.path("total_tokens"));
		if (totalTokens == null && promptTokens != null && completionTokens != null) {
			totalTokens = promptTokens + completionTokens;
		}
		Integer cachedTokens = intOrNull(usage.path("input_tokens_details").path("cached_tokens"));
		Integer reasoningTokens = intOrNull(usage.path("output_tokens_details").path("reasoning_tokens"));
		Usage.PromptTokensDetails promptDetails = cachedTokens != null
				? new Usage.PromptTokensDetails(null, cachedTokens) : null;
		Usage.CompletionTokenDetails completionDetails = reasoningTokens != null
				? new Usage.CompletionTokenDetails(reasoningTokens, null, null, null) : null;
		return new Usage(completionTokens, promptTokens, totalTokens, promptDetails, completionDetails,
				longOrNull(usage, response, "compute_millis"), longOrNull(usage, response, "wait_millis"),
				textOrNull(usage, response, "price"), textOrNull(usage, response, "price_with_discount"));
	}

	private static Integer intOrNull(JsonNode node) {
		return node.isNumber() ? node.asInt() : null;
	}

	private static Long longOrNull(JsonNode primary, JsonNode fallback, String field) {
		JsonNode node = primary.path(field);
		if (!node.isNumber()) {
			node = fallback.path(field);
		}
		return node.isNumber() ? node.asLong() : null;
	}

	private static String textOrNull(JsonNode primary, JsonNode fallback, String field) {
		JsonNode node = primary.path(field);
		if (node.isMissingNode() || node.isNull()) {
			node = fallback.path(field);
		}
		return node.isValueNode() && !node.isNull() ? node.asText() : null;
	}

	/** Chat streams carry the role on the first delta only; later deltas leave it out. */
	private Role roleForDelta() {
		if (this.roleSent) {
			return null;
		}
		this.roleSent = true;
		return Role.ASSISTANT;
	}

	private ChatCompletionChunk chunk(List<ChunkChoice> choices, Usage usage) {
		return new ChatCompletionChunk(this.responseId, choices, this.createdAt, this.model, null, null, CHUNK_OBJECT,
				usage);
	}

}
