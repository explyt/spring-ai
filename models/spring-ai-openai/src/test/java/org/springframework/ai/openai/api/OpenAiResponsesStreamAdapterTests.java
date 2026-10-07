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

import java.util.List;

import org.junit.jupiter.api.Test;
import reactor.core.publisher.Flux;

import org.springframework.ai.openai.api.OpenAiApi.ChatCompletionChunk;
import org.springframework.ai.openai.api.OpenAiApi.ChatCompletionFinishReason;
import org.springframework.ai.openai.api.OpenAiApi.ChatCompletionMessage.Role;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * Unit tests for {@link OpenAiResponsesStreamAdapter}: Responses API SSE events must be
 * translated into the Chat Completions chunk shape the typed pipeline understands.
 */
class OpenAiResponsesStreamAdapterTests {

	private static final String CREATED = "{\"type\":\"response.created\",\"sequence_number\":0,"
			+ "\"response\":{\"id\":\"resp_1\",\"object\":\"response\",\"created_at\":1700000000,"
			+ "\"model\":\"gpt-test\",\"status\":\"in_progress\",\"output\":[]}}";

	private static final String TEXT_DELTA_1 = "{\"type\":\"response.output_text.delta\",\"sequence_number\":3,"
			+ "\"item_id\":\"msg_1\",\"output_index\":0,\"content_index\":0,\"delta\":\"Hel\"}";

	private static final String TEXT_DELTA_2 = "{\"type\":\"response.output_text.delta\",\"sequence_number\":4,"
			+ "\"item_id\":\"msg_1\",\"output_index\":0,\"content_index\":0,\"delta\":\"lo\"}";

	private static final String REASONING_DELTA = "{\"type\":\"response.reasoning_summary_text.delta\","
			+ "\"sequence_number\":2,\"item_id\":\"rs_1\",\"output_index\":0,\"summary_index\":0,\"delta\":\"thinking\"}";

	private static final String FN_ADDED = "{\"type\":\"response.output_item.added\",\"sequence_number\":5,"
			+ "\"output_index\":1,\"item\":{\"id\":\"fc_1\",\"type\":\"function_call\",\"status\":\"in_progress\","
			+ "\"call_id\":\"call_abc\",\"name\":\"shell\",\"arguments\":\"\"}}";

	private static final String FN_ARGS_DELTA = "{\"type\":\"response.function_call_arguments.delta\","
			+ "\"sequence_number\":6,\"item_id\":\"fc_1\",\"output_index\":1,\"delta\":\"{\\\"cmd\\\":\"}";

	private static final String FN_DONE = "{\"type\":\"response.output_item.done\",\"sequence_number\":8,"
			+ "\"output_index\":1,\"item\":{\"id\":\"fc_1\",\"type\":\"function_call\",\"status\":\"completed\","
			+ "\"call_id\":\"call_abc\",\"name\":\"shell\",\"arguments\":\"{\\\"cmd\\\":\\\"ls\\\"}\"}}";

	private static final String COMPLETED = "{\"type\":\"response.completed\",\"sequence_number\":9,"
			+ "\"response\":{\"id\":\"resp_1\",\"object\":\"response\",\"created_at\":1700000000,"
			+ "\"model\":\"gpt-test\",\"status\":\"completed\",\"output\":[],"
			+ "\"usage\":{\"input_tokens\":9,\"input_tokens_details\":{\"cached_tokens\":4},"
			+ "\"output_tokens\":12,\"output_tokens_details\":{\"reasoning_tokens\":5},\"total_tokens\":21,"
			+ "\"compute_millis\":321,\"price\":\"0.001\",\"price_with_discount\":\"0.0005\"}}}";

	private static final String INCOMPLETE = "{\"type\":\"response.incomplete\",\"sequence_number\":9,"
			+ "\"response\":{\"id\":\"resp_1\",\"model\":\"gpt-test\",\"status\":\"incomplete\","
			+ "\"incomplete_details\":{\"reason\":\"max_output_tokens\"},"
			+ "\"usage\":{\"input_tokens\":1,\"output_tokens\":2,\"total_tokens\":3}}}";

	private static final String FAILED = "{\"type\":\"response.failed\",\"sequence_number\":9,"
			+ "\"response\":{\"id\":\"resp_1\",\"status\":\"failed\","
			+ "\"error\":{\"code\":\"server_error\",\"message\":\"boom\"}}}";

	private static List<ChatCompletionChunk> adapt(String... frames) {
		return OpenAiResponsesStreamAdapter.toChatCompletionChunks(Flux.just(frames)).collectList().block();
	}

	@Test
	void textDeltasBecomeContentChunksWithRoleOnFirstOnly() {
		List<ChatCompletionChunk> chunks = adapt(CREATED, TEXT_DELTA_1, TEXT_DELTA_2);

		assertThat(chunks).hasSize(2);
		assertThat(chunks).allSatisfy(chunk -> {
			assertThat(chunk.id()).isEqualTo("resp_1");
			assertThat(chunk.model()).isEqualTo("gpt-test");
			assertThat(chunk.created()).isEqualTo(1700000000L);
			assertThat(chunk.object()).isEqualTo("chat.completion.chunk");
			assertThat(chunk.usage()).isNull();
		});
		assertThat(chunks.get(0).choices().get(0).delta().content()).isEqualTo("Hel");
		assertThat(chunks.get(0).choices().get(0).delta().role()).isEqualTo(Role.ASSISTANT);
		assertThat(chunks.get(1).choices().get(0).delta().content()).isEqualTo("lo");
		assertThat(chunks.get(1).choices().get(0).delta().role()).isNull();
	}

	@Test
	void reasoningSummaryDeltaBecomesReasoningContent() {
		List<ChatCompletionChunk> chunks = adapt(CREATED, REASONING_DELTA);

		assertThat(chunks).hasSize(1);
		assertThat(chunks.get(0).choices().get(0).delta().getReasoningContent()).isEqualTo("thinking");
		assertThat(chunks.get(0).choices().get(0).delta().content()).isNull();
	}

	@Test
	void functionCallIsEmittedOnceCompleteAndFinishesWithToolCalls() {
		List<ChatCompletionChunk> chunks = adapt(CREATED, FN_ADDED, FN_ARGS_DELTA, FN_DONE, COMPLETED);

		// tool call + finish + usage; added/argument deltas produce nothing.
		assertThat(chunks).hasSize(3);

		var toolCalls = chunks.get(0).choices().get(0).delta().toolCalls();
		assertThat(toolCalls).hasSize(1);
		assertThat(toolCalls.get(0).index()).isEqualTo(0);
		assertThat(toolCalls.get(0).id()).isEqualTo("call_abc");
		assertThat(toolCalls.get(0).type()).isEqualTo("function");
		assertThat(toolCalls.get(0).function().name()).isEqualTo("shell");
		assertThat(toolCalls.get(0).function().arguments()).isEqualTo("{\"cmd\":\"ls\"}");

		assertThat(chunks.get(1).choices().get(0).finishReason()).isEqualTo(ChatCompletionFinishReason.TOOL_CALLS);
		assertThat(chunks.get(2).choices()).isEmpty();
		assertThat(chunks.get(2).usage()).isNotNull();
	}

	@Test
	void completedTranslatesUsageIncludingGatewayExtensions() {
		List<ChatCompletionChunk> chunks = adapt(CREATED, TEXT_DELTA_1, COMPLETED);

		assertThat(chunks).hasSize(3);
		assertThat(chunks.get(1).choices().get(0).finishReason()).isEqualTo(ChatCompletionFinishReason.STOP);
		assertThat(chunks.get(1).usage()).isNull();

		OpenAiApi.Usage usage = chunks.get(2).usage();
		assertThat(chunks.get(2).choices()).isEmpty();
		assertThat(usage.promptTokens()).isEqualTo(9);
		assertThat(usage.completionTokens()).isEqualTo(12);
		assertThat(usage.totalTokens()).isEqualTo(21);
		assertThat(usage.promptTokensDetails().cachedTokens()).isEqualTo(4);
		assertThat(usage.completionTokenDetails().reasoningTokens()).isEqualTo(5);
		assertThat(usage.computeMillis()).isEqualTo(321L);
		assertThat(usage.waitMillis()).isNull();
		assertThat(usage.price()).isEqualTo("0.001");
		assertThat(usage.priceWithDiscount()).isEqualTo("0.0005");
	}

	@Test
	void incompleteMaxOutputTokensFinishesWithLength() {
		List<ChatCompletionChunk> chunks = adapt(CREATED, TEXT_DELTA_1, INCOMPLETE);

		assertThat(chunks).hasSize(3);
		assertThat(chunks.get(1).choices().get(0).finishReason()).isEqualTo(ChatCompletionFinishReason.LENGTH);
		assertThat(chunks.get(2).usage().totalTokens()).isEqualTo(3);
	}

	@Test
	void refusalAndRawReasoningTextDeltasAreMapped() {
		List<ChatCompletionChunk> chunks = adapt(CREATED,
				"{\"type\":\"response.refusal.delta\",\"sequence_number\":2,\"delta\":\"I cannot\"}",
				"{\"type\":\"response.reasoning_text.delta\",\"sequence_number\":3,\"delta\":\"raw cot\"}");

		assertThat(chunks).hasSize(2);
		assertThat(chunks.get(0).choices().get(0).delta().refusal()).isEqualTo("I cannot");
		assertThat(chunks.get(0).choices().get(0).delta().content()).isNull();
		assertThat(chunks.get(1).choices().get(0).delta().getReasoningContent()).isEqualTo("raw cot");
	}

	@Test
	void incompleteContentFilterFinishesWithContentFilter() {
		String incomplete = "{\"type\":\"response.incomplete\",\"sequence_number\":9,"
				+ "\"response\":{\"id\":\"resp_1\",\"model\":\"gpt-test\",\"status\":\"incomplete\","
				+ "\"incomplete_details\":{\"reason\":\"content_filter\"}}}";
		List<ChatCompletionChunk> chunks = adapt(CREATED, TEXT_DELTA_1, incomplete);

		// No usage object on this terminal: finish chunk only, no usage chunk.
		assertThat(chunks).hasSize(2);
		assertThat(chunks.get(1).choices().get(0).finishReason()).isEqualTo(ChatCompletionFinishReason.CONTENT_FILTER);
	}

	@Test
	void failedAndUnknownEventsAndDoneSentinelProduceNothing() {
		List<ChatCompletionChunk> chunks = adapt(CREATED, "{\"type\":\"response.some_future_event\"}",
				"{\"type\":\"error\",\"code\":\"rate_limit\",\"message\":\"slow down\"}", FAILED, "not json", "[DONE]");

		assertThat(chunks).isEmpty();
	}

}
