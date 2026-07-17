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

import io.micrometer.observation.ObservationRegistry;
import org.junit.jupiter.api.Test;
import reactor.core.publisher.Flux;

import org.springframework.ai.chat.metadata.Usage;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.chat.model.DualStreamItem;
import org.springframework.ai.chat.prompt.Prompt;
import org.springframework.ai.model.tool.ToolCallingManager;
import org.springframework.ai.openai.OpenAiChatModel;
import org.springframework.ai.openai.OpenAiChatOptions;
import org.springframework.ai.openai.api.OpenAiApi;
import org.springframework.ai.retry.RetryUtils;
import org.springframework.http.codec.ServerSentEvent;
import org.springframework.retry.support.RetryTemplate;
import org.springframework.util.StringUtils;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.doReturn;
import static org.mockito.Mockito.spy;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;

/**
 * Unit tests for the OpenAI raw-response passthrough tee
 * ({@code OpenAiChatModel#streamRawPassthrough}): a single upstream SSE stream must be
 * emitted BOTH as verbatim {@link DualStreamItem.RawFrame}s (including unknown fields,
 * the usage frame and the {@code [DONE]} sentinel) AND as
 * {@link DualStreamItem.TypedChunk}s with correctly accumulated usage.
 */
public class OpenAiChatModelStreamRawPassthroughTests {

	// A content chunk carrying a vendor-extension field that the typed DTO does not
	// know about: it must survive verbatim in the raw branch.
	private static final String CONTENT_CHUNK = "{\"id\":\"id-1\",\"object\":\"chat.completion.chunk\","
			+ "\"created\":666,\"model\":\"gpt-test\",\"choices\":[{\"index\":0,"
			+ "\"delta\":{\"role\":\"assistant\",\"content\":\"Hello\"}}],\"x_vendor_extension\":\"keep-me\"}";

	// The final usage-only chunk OpenAI emits when stream_options.include_usage=true.
	private static final String USAGE_CHUNK = "{\"id\":\"id-1\",\"object\":\"chat.completion.chunk\","
			+ "\"created\":666,\"model\":\"gpt-test\",\"choices\":[],"
			+ "\"usage\":{\"completion_tokens\":12,\"prompt_tokens\":9,\"total_tokens\":21}}";

	private static final String DONE = "[DONE]";

	private OpenAiChatModel chatModel;

	private OpenAiApi apiSpy;

	private void setupChatModel(Flux<ServerSentEvent<String>> sseFrames) {
		OpenAiApi realApi = OpenAiApi.builder().apiKey("test").build();
		this.apiSpy = spy(realApi);
		doReturn(sseFrames).when(this.apiSpy).chatCompletionStreamRawSse(anyString(), any());
		RetryTemplate retryTemplate = RetryUtils.DEFAULT_RETRY_TEMPLATE;
		this.chatModel = new OpenAiChatModel(this.apiSpy, OpenAiChatOptions.builder().build(),
				ToolCallingManager.builder().build(), retryTemplate, ObservationRegistry.NOOP);
	}

	private static ServerSentEvent<String> sse(String data) {
		// OpenAI chat SSE has no event names: event() stays null, like on the wire.
		return ServerSentEvent.builder(data).build();
	}

	@Test
	void teeEmitsVerbatimRawFramesAndTypedChunksWithUsage() {
		setupChatModel(Flux.just(sse(CONTENT_CHUNK), sse(USAGE_CHUNK), sse(DONE)));

		List<DualStreamItem> items = this.chatModel.streamRawPassthrough(new Prompt("test"), "{\"messages\":[]}")
			.collectList()
			.block();

		assertThat(items).isNotEmpty();

		// RAW branch: every frame arrives verbatim, in order, with null event names,
		// including the unknown vendor field and the [DONE] sentinel.
		List<DualStreamItem.RawFrame> rawFrames = items.stream()
			.filter(DualStreamItem.RawFrame.class::isInstance)
			.map(DualStreamItem.RawFrame.class::cast)
			.toList();
		assertThat(rawFrames).hasSize(3);
		assertThat(rawFrames.get(0).data()).isEqualTo(CONTENT_CHUNK);
		assertThat(rawFrames.get(0).data()).contains("\"x_vendor_extension\":\"keep-me\"");
		assertThat(rawFrames.get(1).data()).isEqualTo(USAGE_CHUNK);
		assertThat(rawFrames.get(2).data()).isEqualTo(DONE);
		assertThat(rawFrames).allSatisfy(frame -> assertThat(frame.event()).isNull());

		// TYPED branch: the same pipeline as internalStream, so the usage from the
		// final chunk is merged onto the content response (raw mode forces the gate).
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

		Usage usage = contentResponse.getMetadata().getUsage();
		assertThat(usage).isNotNull();
		assertThat(usage.getPromptTokens()).isEqualTo(9);
		assertThat(usage.getCompletionTokens()).isEqualTo(12);
		assertThat(usage.getTotalTokens()).isEqualTo(21);
	}

	@Test
	void makesExactlyOneHttpRequest() {
		setupChatModel(Flux.just(sse(CONTENT_CHUNK), sse(USAGE_CHUNK), sse(DONE)));

		this.chatModel.streamRawPassthrough(new Prompt("test"), "{\"messages\":[]}").collectList().block();

		verify(this.apiSpy, times(1)).chatCompletionStreamRawSse(anyString(), any());
	}

}
