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

import java.io.IOException;
import java.util.List;
import java.util.Objects;
import java.util.stream.Collectors;

import okhttp3.mockwebserver.MockWebServer;
import okhttp3.mockwebserver.RecordedRequest;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import tools.jackson.databind.JsonNode;

import org.springframework.ai.openai.api.OpenAiApi.ChatCompletion;
import org.springframework.ai.openai.api.OpenAiApi.ChatCompletionChunk;
import org.springframework.ai.openai.api.OpenAiApi.ChatCompletionFinishReason;
import org.springframework.ai.openai.api.OpenAiApi.ChatCompletionMessage;
import org.springframework.ai.openai.api.OpenAiApi.ChatCompletionMessage.Role;
import org.springframework.ai.openai.api.OpenAiApi.ChatCompletionMessage.ToolCall;
import org.springframework.ai.openai.api.OpenAiApi.ChatCompletionRequest;
import org.springframework.ai.retry.NonTransientAiException;
import org.springframework.http.HttpHeaders;
import org.springframework.http.ResponseEntity;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.springframework.ai.openai.api.ResponsesApiFixtures.CALL_BERLIN_DONE;
import static org.springframework.ai.openai.api.ResponsesApiFixtures.CALL_PARIS_DONE;
import static org.springframework.ai.openai.api.ResponsesApiFixtures.CHAT_COMPLETION;
import static org.springframework.ai.openai.api.ResponsesApiFixtures.COMPLETED;
import static org.springframework.ai.openai.api.ResponsesApiFixtures.CREATED;
import static org.springframework.ai.openai.api.ResponsesApiFixtures.FAILED;
import static org.springframework.ai.openai.api.ResponsesApiFixtures.RESPONSE_OBJECT;
import static org.springframework.ai.openai.api.ResponsesApiFixtures.TEXT_HEL;
import static org.springframework.ai.openai.api.ResponsesApiFixtures.TEXT_LO;
import static org.springframework.ai.openai.api.ResponsesApiFixtures.body;
import static org.springframework.ai.openai.api.ResponsesApiFixtures.json;
import static org.springframework.ai.openai.api.ResponsesApiFixtures.sse;

class OpenAiApiResponsesWireApiTests {

	private MockWebServer server;

	@BeforeEach
	void setUp() throws IOException {
		this.server = new MockWebServer();
		this.server.start();
	}

	@AfterEach
	void tearDown() throws IOException {
		this.server.shutdown();
	}

	private OpenAiApi.Builder responsesApi() {
		return OpenAiApi.builder()
			.apiKey("test-key")
			.baseUrl(this.server.url("/").toString())
			.wireApi(OpenAiWireApi.RESPONSES);
	}

	private static ChatCompletionRequest request(boolean stream) {
		return new ChatCompletionRequest(List.of(new ChatCompletionMessage("Hi", Role.USER)), "gpt-test", 0.7, stream);
	}

	private static String text(List<ChatCompletionChunk> chunks) {
		return chunks.stream()
			.filter(chunk -> chunk.choices() != null)
			.flatMap(chunk -> chunk.choices().stream())
			.map(choice -> choice.delta().content())
			.filter(Objects::nonNull)
			.collect(Collectors.joining());
	}

	private static int totalTokens(List<ChatCompletionChunk> chunks) {
		return chunks.stream()
			.map(ChatCompletionChunk::usage)
			.filter(Objects::nonNull)
			.reduce((first, second) -> second)
			.orElseThrow()
			.totalTokens();
	}

	@Test
	void streamPostsResponsesRequestAndAdaptsEvents() throws InterruptedException {
		this.server.enqueue(sse(CREATED, TEXT_HEL, TEXT_LO, COMPLETED));

		List<ChatCompletionChunk> chunks = responsesApi().build()
			.chatCompletionStream(request(true).streamOptions(ChatCompletionRequest.StreamOptions.INCLUDE_USAGE))
			.collectList()
			.block();

		RecordedRequest recorded = this.server.takeRequest();
		assertThat(recorded.getPath()).isEqualTo("/v1/responses");
		assertThat(recorded.getHeader(HttpHeaders.AUTHORIZATION)).isEqualTo("Bearer test-key");
		JsonNode sent = body(recorded);
		assertThat(sent.path("stream").asBoolean()).isTrue();
		assertThat(sent.path("store").asBoolean(true)).isFalse();
		assertThat(sent.path("input").get(0).path("content").asText()).isEqualTo("Hi");
		assertThat(text(chunks)).isEqualTo("Hello");
		assertThat(totalTokens(chunks)).isEqualTo(21);
	}

	@Test
	void streamDropsUsageOnlyChunkWhenUsageWasNotRequested() {
		this.server.enqueue(sse(CREATED, TEXT_HEL, TEXT_LO, COMPLETED));

		List<ChatCompletionChunk> chunks = responsesApi().build()
			.chatCompletionStream(request(true))
			.collectList()
			.block();

		assertThat(chunks).isNotEmpty();
		assertThat(text(chunks)).isEqualTo("Hello");
		assertThat(chunks).allSatisfy(chunk -> assertThat(chunk.usage()).isNull());
	}

	@Test
	void streamMergesParallelFunctionCallsIntoOneToolCallsChunk() {
		this.server.enqueue(sse(CREATED, CALL_PARIS_DONE, CALL_BERLIN_DONE, COMPLETED));

		List<ChatCompletionChunk> chunks = responsesApi().build()
			.chatCompletionStream(request(true))
			.collectList()
			.block();

		ChatCompletionChunk toolCalls = chunks.get(0);
		assertThat(toolCalls.choices().get(0).finishReason()).isEqualTo(ChatCompletionFinishReason.TOOL_CALLS);
		assertThat(toolCalls.choices().get(0).delta().toolCalls()).extracting(ToolCall::id)
			.containsExactly("call_1", "call_2");
	}

	@Test
	void streamFailsOnFailedEvent() {
		this.server.enqueue(sse(CREATED, TEXT_HEL, FAILED));

		assertThatThrownBy(() -> responsesApi().build().chatCompletionStream(request(true)).collectList().block())
			.isInstanceOf(NonTransientAiException.class)
			.hasMessageContaining("server_error")
			.hasMessageContaining("boom");
	}

	@Test
	void entityPostsResponsesRequestAndAdaptsResponse() throws InterruptedException {
		this.server.enqueue(json(RESPONSE_OBJECT));

		ResponseEntity<ChatCompletion> entity = responsesApi().build().chatCompletionEntity(request(false));

		RecordedRequest recorded = this.server.takeRequest();
		assertThat(recorded.getPath()).isEqualTo("/v1/responses");
		assertThat(body(recorded).path("stream").asBoolean(true)).isFalse();
		ChatCompletionMessage message = entity.getBody().choices().get(0).message();
		assertThat(message.content()).isEqualTo("Hello");
		assertThat(message.getReasoningContent()).isEqualTo("Plan");
		assertThat(entity.getBody().usage().totalTokens()).isEqualTo(21);
	}

	@Test
	void usesConfiguredResponsesPathAndReasoningSummary() throws InterruptedException {
		this.server.enqueue(json(RESPONSE_OBJECT));

		responsesApi().responsesPath("/api/v4/responses")
			.reasoningSummary("auto")
			.build()
			.chatCompletionEntity(request(false));

		RecordedRequest recorded = this.server.takeRequest();
		assertThat(recorded.getPath()).isEqualTo("/api/v4/responses");
		assertThat(body(recorded).path("reasoning").path("summary").asText()).isEqualTo("auto");
	}

	@Test
	void chatCompletionsRemainsTheDefaultWireApi() throws InterruptedException {
		this.server.enqueue(json(CHAT_COMPLETION));

		OpenAiApi.builder()
			.apiKey("test-key")
			.baseUrl(this.server.url("/").toString())
			.build()
			.chatCompletionEntity(request(false));

		assertThat(this.server.takeRequest().getPath()).isEqualTo("/v1/chat/completions");
	}

	@Test
	void mutateKeepsWireApiAndReasoningSummary() {
		OpenAiApi api = responsesApi().reasoningSummary("auto").build().mutate().build();

		assertThat(api.getWireApi()).isEqualTo(OpenAiWireApi.RESPONSES);
		assertThat(api.getReasoningSummary()).isEqualTo("auto");
	}

}
