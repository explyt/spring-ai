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

import java.io.IOException;
import java.util.List;
import java.util.Objects;
import java.util.function.Function;
import java.util.stream.Collectors;

import okhttp3.mockwebserver.MockWebServer;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import tools.jackson.databind.JsonNode;

import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.chat.prompt.Prompt;
import org.springframework.ai.model.ModelOptionsUtils;
import org.springframework.ai.openai.OpenAiChatModel;
import org.springframework.ai.openai.OpenAiChatOptions;
import org.springframework.ai.openai.api.OpenAiApi;
import org.springframework.ai.openai.api.OpenAiWireApi;
import org.springframework.ai.tool.ToolCallback;
import org.springframework.ai.tool.function.FunctionToolCallback;

import static org.assertj.core.api.Assertions.assertThat;
import static org.springframework.ai.openai.api.ResponsesApiFixtures.CALL_PARIS_DONE;
import static org.springframework.ai.openai.api.ResponsesApiFixtures.COMPLETED;
import static org.springframework.ai.openai.api.ResponsesApiFixtures.CREATED;
import static org.springframework.ai.openai.api.ResponsesApiFixtures.RESPONSE_OBJECT;
import static org.springframework.ai.openai.api.ResponsesApiFixtures.TEXT_HEL;
import static org.springframework.ai.openai.api.ResponsesApiFixtures.TEXT_LO;
import static org.springframework.ai.openai.api.ResponsesApiFixtures.TEXT_SUNNY;
import static org.springframework.ai.openai.api.ResponsesApiFixtures.body;
import static org.springframework.ai.openai.api.ResponsesApiFixtures.json;
import static org.springframework.ai.openai.api.ResponsesApiFixtures.sse;

class OpenAiChatModelResponsesWireApiTests {

	public record WeatherRequest(String city) {
	}

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

	private OpenAiChatModel chatModel(OpenAiChatOptions options) {
		OpenAiApi api = OpenAiApi.builder()
			.apiKey("test-key")
			.baseUrl(this.server.url("/").toString())
			.wireApi(OpenAiWireApi.RESPONSES)
			.reasoningSummary("auto")
			.build();
		return OpenAiChatModel.builder().openAiApi(api).defaultOptions(options).build();
	}

	private static String text(List<ChatResponse> responses) {
		return responses.stream()
			.map(ChatResponse::getResult)
			.filter(Objects::nonNull)
			.map(generation -> generation.getOutput().getText())
			.filter(Objects::nonNull)
			.collect(Collectors.joining());
	}

	@Test
	void streamRunsToolLoopOverResponsesApi() throws InterruptedException {
		this.server.enqueue(sse(CREATED, CALL_PARIS_DONE, COMPLETED));
		this.server.enqueue(sse(CREATED, TEXT_SUNNY, COMPLETED));
		Function<WeatherRequest, String> weather = request -> "Sunny in " + request.city();
		ToolCallback weatherTool = FunctionToolCallback.builder("weather", weather)
			.description("Current weather")
			.inputType(WeatherRequest.class)
			.build();
		OpenAiChatOptions options = OpenAiChatOptions.builder()
			.model("gpt-test")
			.reasoningEffort("low")
			.streamUsage(true)
			.toolCallbacks(weatherTool)
			.build();

		List<ChatResponse> responses = chatModel(options).stream(new Prompt("Weather in Paris?")).collectList().block();

		JsonNode first = body(this.server.takeRequest());
		assertThat((Object) first.path("reasoning"))
			.isEqualTo(ModelOptionsUtils.JSON_MAPPER.readTree("{\"effort\":\"low\",\"summary\":\"auto\"}"));
		JsonNode tool = first.path("tools").get(0);
		assertThat(tool.path("type").asText()).isEqualTo("function");
		assertThat(tool.path("name").asText()).isEqualTo("weather");
		assertThat(tool.has("function")).isFalse();
		JsonNode input = body(this.server.takeRequest()).path("input");
		assertThat(input).extracting(item -> item.path("type").asText())
			.containsExactly("message", "function_call", "function_call_output");
		assertThat(input.get(1).path("call_id").asText()).isEqualTo("call_1");
		assertThat(input.get(1).path("arguments").asText()).isEqualTo("{\"city\":\"Paris\"}");
		assertThat(input.get(2).path("call_id").asText()).isEqualTo("call_1");
		assertThat(input.get(2).path("output").asText()).contains("Sunny in Paris");
		assertThat(text(responses)).isEqualTo("Sunny");
	}

	@Test
	void streamWithoutStreamUsageDoesNotEmitUsageOnlyResponse() {
		this.server.enqueue(sse(CREATED, TEXT_HEL, TEXT_LO, COMPLETED));
		OpenAiChatOptions options = OpenAiChatOptions.builder().model("gpt-test").build();

		List<ChatResponse> responses = chatModel(options).stream(new Prompt("Hi")).collectList().block();

		assertThat(responses).isNotEmpty();
		assertThat(text(responses)).isEqualTo("Hello");
		assertThat(responses).allSatisfy(response -> assertThat(response.getResult()).isNotNull());
	}

	@Test
	void callMapsResponsesObjectToChatResponse() throws InterruptedException {
		this.server.enqueue(json(RESPONSE_OBJECT));

		ChatResponse response = chatModel(OpenAiChatOptions.builder().model("gpt-test").build()).call(new Prompt("Hi"));

		assertThat(response.getResult().getOutput().getText()).isEqualTo("Hello");
		assertThat(response.getResult().getOutput().getReasoningContent()).isEqualTo("Plan");
		assertThat(response.getMetadata().getUsage().getTotalTokens()).isEqualTo(21);
		assertThat(body(this.server.takeRequest()).path("stream").asBoolean(true)).isFalse();
	}

}
