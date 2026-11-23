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

package org.springframework.ai.openai.chat;

import java.util.List;

import tools.jackson.databind.JsonNode;
import io.micrometer.observation.ObservationRegistry;
import org.junit.jupiter.api.Test;
import org.mockito.ArgumentCaptor;
import reactor.core.publisher.Flux;

import org.springframework.ai.chat.metadata.Usage;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.chat.model.DualStreamItem;
import org.springframework.ai.chat.prompt.Prompt;
import org.springframework.ai.model.ModelOptionsUtils;
import org.springframework.ai.model.tool.ToolCallingManager;
import org.springframework.ai.openai.OpenAiChatModel;
import org.springframework.ai.openai.OpenAiChatOptions;
import org.springframework.ai.openai.api.OpenAiApi;
import org.springframework.ai.retry.RetryUtils;
import org.springframework.http.codec.ServerSentEvent;
import org.springframework.core.retry.RetryTemplate;
import org.springframework.util.StringUtils;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.doReturn;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.spy;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;

/**
 * Unit tests for the Responses API raw passthrough tee
 * ({@code OpenAiChatModel#streamRawPassthroughResponses}): one upstream Responses SSE
 * stream must be emitted BOTH as verbatim {@link DualStreamItem.RawFrame}s (event names
 * preserved, unknown fields intact) AND as {@link DualStreamItem.TypedChunk}s carrying
 * the assistant text and the usage from {@code response.completed}; and the forwarded
 * body must carry the gateway overrides.
 */
public class OpenAiChatModelStreamRawPassthroughResponsesTests {

	private static final String CREATED = "{\"type\":\"response.created\",\"sequence_number\":0,"
			+ "\"response\":{\"id\":\"resp_1\",\"object\":\"response\",\"created_at\":1700000000,"
			+ "\"model\":\"gpt-test\",\"status\":\"in_progress\",\"output\":[]}}";

	// Carries a vendor-extension field the adapter does not know about: it must
	// survive verbatim in the raw branch.
	private static final String TEXT_DELTA = "{\"type\":\"response.output_text.delta\",\"sequence_number\":3,"
			+ "\"item_id\":\"msg_1\",\"output_index\":0,\"content_index\":0,\"delta\":\"Hello\","
			+ "\"x_vendor_extension\":\"keep-me\"}";

	private static final String COMPLETED = "{\"type\":\"response.completed\",\"sequence_number\":9,"
			+ "\"response\":{\"id\":\"resp_1\",\"object\":\"response\",\"created_at\":1700000000,"
			+ "\"model\":\"gpt-test\",\"status\":\"completed\",\"output\":[],"
			+ "\"usage\":{\"input_tokens\":9,\"output_tokens\":12,\"total_tokens\":21}}}";

	private OpenAiChatModel chatModel;

	private OpenAiApi apiSpy;

	private void setupChatModel(Flux<ServerSentEvent<String>> sseFrames) {
		OpenAiApi realApi = OpenAiApi.builder().apiKey("test").build();
		this.apiSpy = spy(realApi);
		doReturn(sseFrames).when(this.apiSpy).responsesStreamRawSse(anyString(), any());
		RetryTemplate retryTemplate = RetryUtils.DEFAULT_RETRY_TEMPLATE;
		this.chatModel = new OpenAiChatModel(this.apiSpy, OpenAiChatOptions.builder().model("gpt-upstream").build(),
				ToolCallingManager.builder().build(), retryTemplate, ObservationRegistry.NOOP);
	}

	private static ServerSentEvent<String> sse(String event, String data) {
		return ServerSentEvent.builder(data).event(event).build();
	}

	private Flux<ServerSentEvent<String>> happyStream() {
		return Flux.just(sse("response.created", CREATED), sse("response.output_text.delta", TEXT_DELTA),
				sse("response.completed", COMPLETED));
	}

	@Test
	void teeEmitsVerbatimRawFramesAndTypedChunksWithUsage() {
		setupChatModel(happyStream());

		List<DualStreamItem> items = this.chatModel
			.streamRawPassthroughResponses(new Prompt("test"), "{\"input\":\"hi\"}")
			.collectList()
			.block();

		assertThat(items).isNotEmpty();

		// RAW branch: every frame verbatim, in order, WITH its Responses event name.
		List<DualStreamItem.RawFrame> rawFrames = items.stream()
			.filter(DualStreamItem.RawFrame.class::isInstance)
			.map(DualStreamItem.RawFrame.class::cast)
			.toList();
		assertThat(rawFrames).hasSize(3);
		assertThat(rawFrames.get(0).event()).isEqualTo("response.created");
		assertThat(rawFrames.get(0).data()).isEqualTo(CREATED);
		assertThat(rawFrames.get(1).event()).isEqualTo("response.output_text.delta");
		assertThat(rawFrames.get(1).data()).contains("\"x_vendor_extension\":\"keep-me\"");
		assertThat(rawFrames.get(2).event()).isEqualTo("response.completed");
		assertThat(rawFrames.get(2).data()).isEqualTo(COMPLETED);

		// TYPED branch: the same pipeline as internalStream, so the usage from
		// response.completed is merged onto the content response.
		List<ChatResponse> typedResponses = items.stream()
			.filter(DualStreamItem.TypedChunk.class::isInstance)
			.map(item -> ((DualStreamItem.TypedChunk) item).response())
			.toList();
		assertThat(typedResponses).isNotEmpty();

		ChatResponse contentResponse = typedResponses.stream()
			.filter(r -> r.getResult() != null && StringUtils.hasText(r.getResult().getOutput().getText()))
			.findFirst()
			.orElseThrow(() -> new AssertionError("no typed chunk carried the assistant text"));
		assertThat(contentResponse.getResult().getOutput().getText()).isEqualTo("Hello");

		Usage usage = typedResponses.stream()
			.map(r -> r.getMetadata().getUsage())
			.filter(u -> u != null && u.getTotalTokens() != null && u.getTotalTokens() > 0)
			.findFirst()
			.orElseThrow(() -> new AssertionError("no typed chunk carried usage"));
		assertThat(usage.getPromptTokens()).isEqualTo(9);
		assertThat(usage.getCompletionTokens()).isEqualTo(12);
		assertThat(usage.getTotalTokens()).isEqualTo(21);
	}

	@Test
	void makesExactlyOneHttpRequest() {
		setupChatModel(happyStream());

		this.chatModel.streamRawPassthroughResponses(new Prompt("test"), "{\"input\":\"hi\"}").collectList().block();

		verify(this.apiSpy, times(1)).responsesStreamRawSse(anyString(), any());
	}

	@Test
	void appliesModelStreamAndStoreOverridesAndKeepsEverythingElse() throws Exception {
		setupChatModel(happyStream());

		String rawBody = "{\"model\":\"gpt-client\",\"input\":\"hi\",\"store\":true,\"stream\":false,"
				+ "\"previous_response_id\":null,\"include\":[\"reasoning.encrypted_content\"],"
				+ "\"reasoning\":{\"effort\":\"high\"},\"x_unknown\":1}";

		this.chatModel.streamRawPassthroughResponses(new Prompt("test"), rawBody).collectList().block();

		ArgumentCaptor<String> bodyCaptor = ArgumentCaptor.forClass(String.class);
		verify(this.apiSpy, times(1)).responsesStreamRawSse(bodyCaptor.capture(), any());
		verify(this.apiSpy, never()).chatCompletionStreamRawSse(anyString(), any());

		JsonNode sent = ModelOptionsUtils.JSON_MAPPER.readTree(bodyCaptor.getValue());
		assertThat(sent.get("model").asText()).isEqualTo("gpt-upstream");
		assertThat(sent.get("stream").asBoolean()).isTrue();
		assertThat(sent.get("store").asBoolean()).isFalse();
		assertThat(sent.has("stream_options")).isFalse();
		assertThat(sent.get("include").get(0).asText()).isEqualTo("reasoning.encrypted_content");
		assertThat(sent.get("reasoning").get("effort").asText()).isEqualTo("high");
		assertThat(sent.get("x_unknown").asInt()).isEqualTo(1);
	}

	/**
	 * Regression test for enterprise-team-board#472: a comment-only SSE frame
	 * ({@code : keepalive}, no {@code data:} field, so {@code data()} is null) between
	 * the Responses events used to kill the whole shared stream, because the typed branch
	 * mapped {@code ServerSentEvent::data} with Reactor {@code map}, which throws
	 * {@code NullPointerException("The mapper [...] returned a null value.")} before any
	 * downstream filter can drop the null. The typed branch must skip the frame and the
	 * raw branch must still forward it (as a {@code RawFrame} with null data).
	 */
	@Test
	void commentOnlyKeepaliveFrameDoesNotKillTheStream() {
		ServerSentEvent<String> keepalive = ServerSentEvent.<String>builder().comment("keepalive").build();
		assertThat(keepalive.data()).isNull();
		setupChatModel(Flux.just(sse("response.created", CREATED), keepalive,
				sse("response.output_text.delta", TEXT_DELTA), sse("response.completed", COMPLETED)));

		List<DualStreamItem> items = this.chatModel
			.streamRawPassthroughResponses(new Prompt("test"), "{\"input\":\"hi\"}")
			.collectList()
			.block();

		assertDataLessFrameIsSkippedByTypedAndForwardedByRaw(items, 1);
		assertThat(rawFrames(items).get(1).event()).isNull();
	}

	/**
	 * Regression test for enterprise-team-board#472: an event-only SSE frame
	 * ({@code event: ping} with no {@code data:} field, so {@code data()} is null) used
	 * to kill the whole shared stream through the Reactor {@code map} null-mapper
	 * {@code NullPointerException} in the typed branch. The typed branch must skip the
	 * frame and the raw branch must still forward it with its event name.
	 */
	@Test
	void eventOnlyFrameWithoutDataDoesNotKillTheStream() {
		ServerSentEvent<String> ping = ServerSentEvent.<String>builder().event("ping").build();
		assertThat(ping.data()).isNull();
		setupChatModel(Flux.just(sse("response.created", CREATED), ping, sse("response.output_text.delta", TEXT_DELTA),
				sse("response.completed", COMPLETED)));

		List<DualStreamItem> items = this.chatModel
			.streamRawPassthroughResponses(new Prompt("test"), "{\"input\":\"hi\"}")
			.collectList()
			.block();

		assertDataLessFrameIsSkippedByTypedAndForwardedByRaw(items, 1);
		assertThat(rawFrames(items).get(1).event()).isEqualTo("ping");
	}

	private static List<DualStreamItem.RawFrame> rawFrames(List<DualStreamItem> items) {
		return items.stream()
			.filter(DualStreamItem.RawFrame.class::isInstance)
			.map(DualStreamItem.RawFrame.class::cast)
			.toList();
	}

	/**
	 * Asserts the stream completed (we got here, so {@code block()} did not throw), the
	 * RAW branch forwarded every frame in order - the data-less one at
	 * {@code dataLessIndex} as a {@code RawFrame} with null data, which is the current,
	 * unchanged behaviour of the raw branch - and the TYPED branch still produced the
	 * same content and usage as the happy-path test.
	 */
	private static void assertDataLessFrameIsSkippedByTypedAndForwardedByRaw(List<DualStreamItem> items,
			int dataLessIndex) {
		assertThat(items).isNotEmpty();

		List<DualStreamItem.RawFrame> rawFrames = rawFrames(items);
		assertThat(rawFrames).hasSize(4);
		assertThat(rawFrames.get(dataLessIndex).data()).isNull();
		List<DualStreamItem.RawFrame> dataBearing = rawFrames.stream().filter(f -> f.data() != null).toList();
		assertThat(dataBearing).extracting(DualStreamItem.RawFrame::event)
			.containsExactly("response.created", "response.output_text.delta", "response.completed");
		assertThat(dataBearing).extracting(DualStreamItem.RawFrame::data)
			.containsExactly(CREATED, TEXT_DELTA, COMPLETED);

		List<ChatResponse> typedResponses = items.stream()
			.filter(DualStreamItem.TypedChunk.class::isInstance)
			.map(item -> ((DualStreamItem.TypedChunk) item).response())
			.toList();
		assertThat(typedResponses).isNotEmpty();

		ChatResponse contentResponse = typedResponses.stream()
			.filter(r -> r.getResult() != null && StringUtils.hasText(r.getResult().getOutput().getText()))
			.findFirst()
			.orElseThrow(() -> new AssertionError("no typed chunk carried the assistant text"));
		assertThat(contentResponse.getResult().getOutput().getText()).isEqualTo("Hello");

		Usage usage = typedResponses.stream()
			.map(r -> r.getMetadata().getUsage())
			.filter(u -> u != null && u.getTotalTokens() != null && u.getTotalTokens() > 0)
			.findFirst()
			.orElseThrow(() -> new AssertionError("no typed chunk carried usage"));
		assertThat(usage.getPromptTokens()).isEqualTo(9);
		assertThat(usage.getCompletionTokens()).isEqualTo(12);
		assertThat(usage.getTotalTokens()).isEqualTo(21);
	}

}
