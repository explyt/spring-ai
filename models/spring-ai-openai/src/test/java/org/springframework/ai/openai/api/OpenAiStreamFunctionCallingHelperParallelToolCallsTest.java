/*
 * Copyright 2023-2026 the original author or authors.
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

import java.util.Arrays;
import java.util.List;

import org.junit.jupiter.api.Test;
import reactor.core.publisher.Flux;

import org.springframework.ai.openai.api.OpenAiApi.ChatCompletionChunk;
import org.springframework.ai.openai.api.OpenAiApi.ChatCompletionChunk.ChunkChoice;
import org.springframework.ai.openai.api.OpenAiApi.ChatCompletionFinishReason;
import org.springframework.ai.openai.api.OpenAiApi.ChatCompletionMessage;
import org.springframework.ai.openai.api.OpenAiApi.ChatCompletionMessage.ChatCompletionFunction;
import org.springframework.ai.openai.api.OpenAiApi.ChatCompletionMessage.ToolCall;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.assertj.core.api.Assertions.tuple;

/**
 * Merging of streamed parallel tool calls in {@link OpenAiStreamFunctionCallingHelper},
 * including upstreams (some vLLM tool parsers) that put several tool-call fragments into
 * a single {@code delta.tool_calls}. Deltas with a single fragment must keep the legacy
 * behaviour exactly; the {@code pinsLegacy*} tests fix it on purpose.
 */
class OpenAiStreamFunctionCallingHelperParallelToolCallsTest {

	private final OpenAiStreamFunctionCallingHelper helper = new OpenAiStreamFunctionCallingHelper();

	@Test
	void mergesSeveralToolCallsBatchedIntoOneDelta() {
		// The batched delta is not the first chunk of the window, so it really goes
		// through the message merge (the first chunk is taken as is).
		var merged = mergeAll(chunk(toolCall(0, "call_a", "read_file", "")),
				chunk(toolCall(0, null, null, "{\"path\":\"a\"}"),
						toolCall(1, "call_b", "read_file", "{\"path\":\"b\"}")),
				finish());

		assertThat(toolCallsOf(merged))
			.extracting(ToolCall::index, ToolCall::id, t -> t.function().name(), t -> t.function().arguments())
			.containsExactly(tuple(0, "call_a", "read_file", "{\"path\":\"a\"}"),
					tuple(1, "call_b", "read_file", "{\"path\":\"b\"}"));
	}

	@Test
	void routesInterleavedArgumentFragmentsByIndex() {
		var merged = mergeAll(chunk(toolCall(0, "call_a", "read_file", ""), toolCall(1, "call_b", "grep", "")),
				chunk(toolCall(0, null, null, "{\"path\":"), toolCall(1, null, null, "{\"q\":")),
				chunk(toolCall(1, null, null, "\"x\"}"), toolCall(0, null, null, "\"a\"}")), finish());

		assertThat(toolCallsOf(merged))
			.extracting(ToolCall::id, t -> t.function().name(), t -> t.function().arguments())
			.containsExactly(tuple("call_a", "read_file", "{\"path\":\"a\"}"),
					tuple("call_b", "grep", "{\"q\":\"x\"}"));
	}

	@Test
	void keepsOneToolCallPerChunkStreamingWorking() {
		var merged = mergeAll(chunk(toolCall(0, "call_a", "read_file", "")), chunk(toolCall(0, null, null, "{}")),
				chunk(toolCall(1, "call_b", "grep", "")), chunk(toolCall(1, null, null, "{\"q\":1}")), finish());

		assertThat(toolCallsOf(merged)).extracting(ToolCall::id, t -> t.function().arguments())
			.containsExactly(tuple("call_a", "{}"), tuple("call_b", "{\"q\":1}"));
	}

	@Test
	void startsNewToolCallWhenUpstreamReusesIndexWithNewId() {
		var merged = mergeAll(chunk(toolCall(0, "call_a", "read_file", "{}")), chunk(toolCall(0, "call_b", "grep", "")),
				chunk(toolCall(0, null, null, "{\"q\":1}")), finish());

		assertThat(toolCallsOf(merged)).extracting(ToolCall::id, t -> t.function().arguments())
			.containsExactly(tuple("call_a", "{}"), tuple("call_b", "{\"q\":1}"));
	}

	@Test
	void keepsIdBasedBehaviourWhenUpstreamSendsNoIndex() {
		var merged = mergeAll(chunk(toolCall(null, "call_a", "read_file", "{\"pa")),
				chunk(toolCall(null, null, null, "th\":1}")), chunk(toolCall(null, "call_b", "grep", "{}")), finish());

		assertThat(toolCallsOf(merged)).extracting(ToolCall::id, t -> t.function().arguments())
			.containsExactly(tuple("call_a", "{\"path\":1}"), tuple("call_b", "{}"));
	}

	@Test
	void rejectsBatchWhenSeveralToolCallsLostTheirIndex() {
		// The legacy single-fragment merge drops the index of a continued call. With two
		// such calls the owner of an id-less fragment is unknown, so it is rejected as
		// before instead of being guessed.
		assertThatThrownBy(() -> mergeAll(chunk(toolCall(0, "call_a", "read_file", "")),
				chunk(toolCall(0, null, null, "{\"path\":")), chunk(toolCall(1, "call_b", "grep", "")),
				chunk(toolCall(1, null, null, "{\"q\":")),
				chunk(toolCall(0, null, null, "\"a\"}"), toolCall(1, null, null, "\"x\"}"))))
			.isInstanceOf(IllegalStateException.class)
			.hasMessageContaining("Cannot attribute");
	}

	@Test
	void rejectsBatchedFragmentWithIdOfToolCallThatLostItsIndex() {
		// Same id, unknown index: a continuation and a new call reusing the id look
		// alike.
		assertThatThrownBy(
				() -> mergeAll(chunk(toolCall(0, "call_a", "read_file", "")), chunk(toolCall(0, null, null, "{\"p\":")),
						chunk(toolCall(0, "call_a", null, "1}"), toolCall(1, "call_b", "grep", "{}"))))
			.isInstanceOf(IllegalStateException.class)
			.hasMessageContaining("Cannot attribute");
	}

	@Test
	void routesBatchedFragmentsByIndexEvenWhenIndexIsNotThePosition() {
		var merged = mergeAll(chunk(toolCall(5, "call_a", "read_file", "")), chunk(toolCall(5, null, null, "{\"p\":")),
				chunk(toolCall(7, "call_b", "grep", "")),
				chunk(toolCall(7, null, null, "{\"q\":2}"), toolCall(5, null, null, "1}")), finish());

		assertThat(toolCallsOf(merged)).extracting(ToolCall::id, t -> t.function().arguments())
			.containsExactly(tuple("call_a", "{\"p\":1}"), tuple("call_b", "{\"q\":2}"));
	}

	@Test
	void routesBatchedFragmentsWhenUpstreamReusesIndexZeroForEveryCall() {
		var merged = mergeAll(chunk(toolCall(0, "call_a", "read_file", "{\"p\":")),
				chunk(toolCall(0, "call_a", null, "1}"), toolCall(0, "call_b", "grep", "{\"q\":")),
				chunk(toolCall(0, null, null, "2}"), toolCall(0, "call_c", "ls", "{}")), finish());

		assertThat(toolCallsOf(merged))
			.extracting(ToolCall::id, t -> t.function().name(), t -> t.function().arguments())
			.containsExactly(tuple("call_a", "read_file", "{\"p\":1}"), tuple("call_b", "grep", "{\"q\":2}"),
					tuple("call_c", "ls", "{}"));
	}

	@Test
	void startsNewToolCallForRepeatedIdUnderNewIndexInBatch() {
		// Some upstreams take ids from the model output, so an id may repeat across
		// parallel calls; a new index is a new call and must not overwrite the first one.
		var merged = mergeAll(chunk(toolCall(0, "call_a", "read_file", "")),
				chunk(toolCall(0, null, null, "{\"p\":1}"), toolCall(1, "call_a", "grep", "{\"q\":2}")), finish());

		assertThat(toolCallsOf(merged))
			.extracting(ToolCall::index, ToolCall::id, t -> t.function().name(), t -> t.function().arguments())
			.containsExactly(tuple(0, "call_a", "read_file", "{\"p\":1}"), tuple(1, "call_a", "grep", "{\"q\":2}"));
	}

	@Test
	void routesSameIndexPairInLaterBatch() {
		// vLLM hermes-style pair: the call opening and its arguments share one delta.
		var merged = mergeAll(chunk(toolCall(0, "call_a", "read_file", "{}")),
				chunk(toolCall(1, "call_b", "grep", ""), toolCall(1, null, null, "{\"q\":2}")), finish());

		assertThat(toolCallsOf(merged))
			.extracting(ToolCall::id, t -> t.function().name(), t -> t.function().arguments())
			.containsExactly(tuple("call_a", "read_file", "{}"), tuple("call_b", "grep", "{\"q\":2}"));
	}

	@Test
	void mergesBatchWithoutIndexById() {
		var merged = mergeAll(chunk(toolCall(null, "call_a", "read_file", "{\"p\":")),
				chunk(toolCall(null, null, null, "1}"), toolCall(null, "call_b", "grep", "{}")), finish());

		assertThat(toolCallsOf(merged)).extracting(ToolCall::id, t -> t.function().arguments())
			.containsExactly(tuple("call_a", "{\"p\":1}"), tuple("call_b", "{}"));
	}

	@Test
	void storesNewBatchedToolCallWithoutFunctionAsEmptyFunction() {
		var merged = mergeAll(chunk(toolCall(0, "call_a", "read_file", "{}")),
				chunk(toolCall(0, null, null, ""), new ToolCall(1, "call_b", "function", null)), finish());

		assertThat(toolCallsOf(merged)).extracting(ToolCall::id, t -> t.function() != null)
			.containsExactly(tuple("call_a", true), tuple("call_b", true));
	}

	@Test
	void keepsAccumulatedFunctionWhenBatchedFragmentHasNone() {
		var merged = mergeAll(chunk(toolCall(0, "call_a", "read_file", "{}")),
				chunk(new ToolCall(0, null, "function", null), toolCall(1, "call_b", "grep", "{}")), finish());

		assertThat(toolCallsOf(merged))
			.extracting(ToolCall::id, t -> t.function().name(), t -> t.function().arguments())
			.containsExactly(tuple("call_a", "read_file", "{}"), tuple("call_b", "grep", "{}"));
	}

	@Test
	void rejectsBatchedFragmentThatCannotBeAttributedToAnyToolCall() {
		// A new index without an id: attaching it to some call would silently corrupt it.
		assertThatThrownBy(() -> mergeAll(chunk(toolCall(0, "call_a", "read_file", "")),
				chunk(toolCall(0, null, null, "{}"), toolCall(1, null, null, "{\"q\":2}"))))
			.isInstanceOf(IllegalStateException.class)
			.hasMessageContaining("Cannot attribute");
	}

	@Test
	void pinsLegacySingleFragmentMergeDroppingIndex() {
		var merged = mergeAll(chunk(toolCall(3, "call_a", "read_file", "{\"pa")),
				chunk(toolCall(3, null, null, "th\":1}")), finish());

		assertThat(toolCallsOf(merged)).extracting(ToolCall::index, ToolCall::id, t -> t.function().arguments())
			.containsExactly(tuple(null, "call_a", "{\"path\":1}"));
	}

	@Test
	void pinsLegacySingleFragmentAfterBatchGoingToLastToolCall() {
		// A single id-less fragment always continues the last call, whatever its index.
		// Kept as is: the legacy path cannot tell this apart from ordinary streaming.
		var merged = mergeAll(chunk(toolCall(0, "call_a", "read_file", "")),
				chunk(toolCall(0, null, null, "{\"p\":"), toolCall(1, "call_b", "grep", "{\"q\":1}")),
				chunk(toolCall(0, null, null, "2}")), finish());

		assertThat(toolCallsOf(merged)).extracting(ToolCall::id, t -> t.function().arguments())
			.containsExactly(tuple("call_a", "{\"p\":"), tuple("call_b", "{\"q\":1}2}"));
	}

	@Test
	void pinsLegacyFirstChunkOfWindowTakenAsIs() {
		// The first chunk of a window is never merged, so a batch there stays as sent;
		// ai-router merges name-less parts afterwards.
		var merged = mergeAll(chunk(toolCall(0, "call_a", "read_file", ""), toolCall(0, null, null, "{}")), finish());

		assertThat(toolCallsOf(merged)).extracting(ToolCall::id, t -> t.function().arguments())
			.containsExactly(tuple("call_a", ""), tuple(null, "{}"));
	}

	@Test
	void pinsLegacyMergeForSingleFragmentDeltas() {
		// Only deltas with several fragments use the new merge: a repeated id in
		// single-fragment deltas still starts a new tool call, as before.
		var merged = mergeAll(chunk(toolCall(null, "call_a", "read_file", "{\"pa")),
				chunk(toolCall(null, "call_a", null, "th\":1}")), finish());

		assertThat(toolCallsOf(merged)).extracting(ToolCall::id, t -> t.function().arguments())
			.containsExactly(tuple("call_a", "{\"pa"), tuple("call_a", "th\":1}"));
	}

	/**
	 * End-to-end over the real SSE parsing pipeline, with frames shaped like the vLLM
	 * output that used to fail with "Currently only one tool call is supported per
	 * message!".
	 */
	@Test
	void parseChatCompletionChunksMergesBatchedParallelToolCalls() {
		OpenAiApi api = OpenAiApi.builder().apiKey("test").build();
		Flux<String> sse = Flux.just(
				"{\"id\":\"r1\",\"object\":\"chat.completion.chunk\",\"model\":\"glm\",\"choices\":[{\"index\":0,"
						+ "\"delta\":{\"role\":\"assistant\",\"content\":\"\"}}]}",
				"{\"id\":\"r1\",\"object\":\"chat.completion.chunk\",\"model\":\"glm\",\"choices\":[{\"index\":0,"
						+ "\"delta\":{\"tool_calls\":["
						+ "{\"index\":0,\"id\":\"call_a\",\"type\":\"function\",\"function\":{\"name\":\"read_file\",\"arguments\":\"\"}}"
						+ "]}}]}",
				"{\"id\":\"r1\",\"object\":\"chat.completion.chunk\",\"model\":\"glm\",\"choices\":[{\"index\":0,"
						+ "\"delta\":{\"tool_calls\":["
						+ "{\"index\":0,\"function\":{\"arguments\":\"{\\\"path\\\":\\\"a\\\"}\"}},"
						+ "{\"index\":1,\"id\":\"call_b\",\"type\":\"function\",\"function\":{\"name\":\"grep\",\"arguments\":\"{\\\"q\\\":\"}}"
						+ "]}}]}",
				"{\"id\":\"r1\",\"object\":\"chat.completion.chunk\",\"model\":\"glm\",\"choices\":[{\"index\":0,"
						+ "\"delta\":{\"tool_calls\":[{\"index\":1,\"function\":{\"arguments\":\"\\\"x\\\"}\"}}]}}]}",
				"{\"id\":\"r1\",\"object\":\"chat.completion.chunk\",\"model\":\"glm\",\"choices\":[{\"index\":0,"
						+ "\"delta\":{},\"finish_reason\":\"tool_calls\"}]}",
				"[DONE]");

		List<ChatCompletionChunk> chunks = api.parseChatCompletionChunks(sse).collectList().block();

		assertThat(chunks).isNotNull();
		List<ToolCall> toolCalls = chunks.stream()
			.flatMap(c -> c.choices().stream())
			.filter(c -> c.delta() != null && c.delta().toolCalls() != null)
			.flatMap(c -> c.delta().toolCalls().stream())
			.toList();
		assertThat(toolCalls).extracting(ToolCall::id, t -> t.function().name(), t -> t.function().arguments())
			.containsExactly(tuple("call_a", "read_file", "{\"path\":\"a\"}"),
					tuple("call_b", "grep", "{\"q\":\"x\"}"));
		assertThat(chunks.get(chunks.size() - 1).choices().get(0).finishReason())
			.isEqualTo(ChatCompletionFinishReason.TOOL_CALLS);
	}

	private ChatCompletionChunk mergeAll(ChatCompletionChunk... chunks) {
		ChatCompletionChunk acc = new ChatCompletionChunk(null, null, null, null, null, null, null, null);
		for (ChatCompletionChunk chunk : chunks) {
			acc = this.helper.merge(acc, chunk);
		}
		return acc;
	}

	private static List<ToolCall> toolCallsOf(ChatCompletionChunk chunk) {
		assertThat(chunk.choices().get(0).finishReason()).isEqualTo(ChatCompletionFinishReason.TOOL_CALLS);
		return chunk.choices().get(0).delta().toolCalls();
	}

	private static ChatCompletionChunk chunk(ToolCall... toolCalls) {
		var delta = new ChatCompletionMessage(null, null, null, null, Arrays.asList(toolCalls), null, null, null, null,
				null);
		return new ChatCompletionChunk("r1", List.of(new ChunkChoice(null, 0, delta, null)), null, null, null, null,
				null, null);
	}

	private static ChatCompletionChunk finish() {
		var delta = new ChatCompletionMessage(null, null);
		return new ChatCompletionChunk("r1",
				List.of(new ChunkChoice(ChatCompletionFinishReason.TOOL_CALLS, 0, delta, null)), null, null, null, null,
				null, null);
	}

	private static ToolCall toolCall(Integer index, String id, String name, String arguments) {
		return new ToolCall(index, id, id != null ? "function" : null, new ChatCompletionFunction(name, arguments));
	}

}
