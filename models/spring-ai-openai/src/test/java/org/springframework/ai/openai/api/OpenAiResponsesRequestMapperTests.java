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
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import tools.jackson.databind.JsonNode;
import tools.jackson.databind.node.ObjectNode;

import org.springframework.ai.model.ModelOptionsUtils;
import org.springframework.ai.openai.api.OpenAiApi.ChatCompletionMessage;
import org.springframework.ai.openai.api.OpenAiApi.ChatCompletionMessage.MediaContent;
import org.springframework.ai.openai.api.OpenAiApi.ChatCompletionMessage.Role;
import org.springframework.ai.openai.api.OpenAiApi.ChatCompletionRequest;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

class OpenAiResponsesRequestMapperTests {

	private static ObjectNode map(String chatCompletionsJson) {
		return map(chatCompletionsJson, null);
	}

	private static ObjectNode map(String chatCompletionsJson, String reasoningSummary) {
		ChatCompletionRequest request = ModelOptionsUtils.jsonToObject(chatCompletionsJson,
				ChatCompletionRequest.class);
		return OpenAiResponsesRequestMapper.toResponsesRequest(request, reasoningSummary);
	}

	private static void assertJson(JsonNode actual, String expected) {
		assertThat((Object) actual).isEqualTo(ModelOptionsUtils.JSON_MAPPER.readTree(expected));
	}

	@Test
	void mapsConversationToInputItemsInOrder() {
		ObjectNode body = map(
				"""
						{"model":"gpt-test","stream":true,"messages":[
						  {"role":"system","content":"Be brief"},
						  {"role":"user","content":"Weather in Paris?"},
						  {"role":"assistant","content":"Checking.","reasoning_content":"hidden","tool_calls":[
						    {"id":"call_1","type":"function","function":{"name":"weather","arguments":"{\\"city\\":\\"Paris\\"}"}}]},
						  {"role":"tool","tool_call_id":"call_1","name":"weather","content":"Sunny"}]}
						""");

		assertJson(body, """
				{"model":"gpt-test","stream":true,"store":false,"input":[
				  {"type":"message","role":"system","content":"Be brief"},
				  {"type":"message","role":"user","content":"Weather in Paris?"},
				  {"type":"message","role":"assistant","content":"Checking."},
				  {"type":"function_call","call_id":"call_1","name":"weather","arguments":"{\\"city\\":\\"Paris\\"}"},
				  {"type":"function_call_output","call_id":"call_1","output":"Sunny"}]}
				""");
	}

	@Test
	void mapsGenerationOptionsAndDropsChatOnlyTransportFields() {
		ObjectNode body = map("""
				{"model":"gpt-test","stream":true,"store":true,"n":1,"stream_options":{"include_usage":true},
				 "temperature":0.2,"top_p":0.9,"max_tokens":512,"parallel_tool_calls":false,
				 "metadata":{"chat":"42"},"user":"u1","prompt_cache_key":"cache-1","service_tier":"flex",
				 "reasoning_effort":"low","verbosity":"low",
				 "messages":[{"role":"user","content":"Hi"}]}
				""");

		assertJson(body, """
				{"model":"gpt-test","stream":true,"store":false,
				 "temperature":0.2,"top_p":0.9,"max_output_tokens":512,"parallel_tool_calls":false,
				 "metadata":{"chat":"42"},"user":"u1","prompt_cache_key":"cache-1","service_tier":"flex",
				 "reasoning":{"effort":"low"},"text":{"verbosity":"low"},
				 "input":[{"type":"message","role":"user","content":"Hi"}]}
				""");
	}

	@Test
	void prefersMaxCompletionTokensOverMaxTokens() {
		ObjectNode body = map("""
				{"model":"gpt-test","stream":false,"max_tokens":100,"max_completion_tokens":200,
				 "messages":[{"role":"user","content":"Hi"}]}
				""");

		assertThat(body.path("max_output_tokens").asInt()).isEqualTo(200);
		assertThat(body.path("stream").asBoolean(true)).isFalse();
	}

	@Test
	void requestsConfiguredReasoningSummaryWithOrWithoutEffort() {
		String withEffort = """
				{"model":"gpt-test","stream":true,"reasoning_effort":"medium","messages":[{"role":"user","content":"Hi"}]}
				""";
		String withoutEffort = """
				{"model":"gpt-test","stream":true,"messages":[{"role":"user","content":"Hi"}]}
				""";

		assertJson(map(withEffort, "auto").path("reasoning"), """
				{"effort":"medium","summary":"auto"}""");
		assertJson(map(withoutEffort, "auto").path("reasoning"), """
				{"summary":"auto"}""");
		assertThat(map(withoutEffort).has("reasoning")).isFalse();
	}

	@Test
	void flattensFunctionToolsAndDefaultsStrictToFalse() {
		ObjectNode body = map(
				"""
						{"model":"gpt-test","stream":true,"tool_choice":"required","messages":[{"role":"user","content":"Hi"}],
						 "tools":[
						   {"type":"function","function":{"name":"weather","description":"Current weather","strict":true,
						     "parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"],"additionalProperties":false}}},
						   {"type":"function","function":{"name":"time","parameters":{"type":"object","properties":{}}}}]}
						""");

		assertJson(body.path("tools"),
				"""
						[{"type":"function","name":"weather","description":"Current weather","strict":true,
						  "parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"],"additionalProperties":false}},
						 {"type":"function","name":"time","strict":false,"parameters":{"type":"object","properties":{}}}]
						""");
		assertThat(body.path("tool_choice").asText()).isEqualTo("required");
	}

	@Test
	void flattensNamedToolChoice() {
		ObjectNode body = map("""
				{"model":"gpt-test","stream":true,"tool_choice":{"type":"function","function":{"name":"weather"}},
				 "messages":[{"role":"user","content":"Hi"}]}
				""");

		assertJson(body.path("tool_choice"), """
				{"type":"function","name":"weather"}""");
	}

	@Test
	void mapsResponseFormatToTextFormat() {
		ObjectNode jsonSchema = map("""
				{"model":"gpt-test","stream":true,"messages":[{"role":"user","content":"Hi"}],
				 "response_format":{"type":"json_schema","json_schema":{"name":"answer","strict":true,
				   "schema":{"type":"object","properties":{"value":{"type":"string"}}}}}}
				""");
		ObjectNode jsonObject = map("""
				{"model":"gpt-test","stream":true,"messages":[{"role":"user","content":"Hi"}],
				 "response_format":{"type":"json_object"}}
				""");

		assertJson(jsonSchema.path("text"), """
				{"format":{"type":"json_schema","name":"answer","strict":true,
				  "schema":{"type":"object","properties":{"value":{"type":"string"}}}}}
				""");
		assertJson(jsonObject.path("text"), """
				{"format":{"type":"json_object"}}""");
	}

	@Test
	void mapsTypedMultimodalUserContentToInputParts() {
		ChatCompletionMessage message = new ChatCompletionMessage(
				List.of(new MediaContent("Describe"),
						new MediaContent(new MediaContent.ImageUrl("data:image/png;base64,AAA")),
						new MediaContent(new MediaContent.InputFile("report.pdf", "data:application/pdf;base64,BBB"))),
				Role.USER);
		ChatCompletionRequest request = new ChatCompletionRequest(List.of(message), "gpt-test", 0.0, true);

		ObjectNode body = OpenAiResponsesRequestMapper.toResponsesRequest(request, null);

		assertJson(body.path("input"), """
				[{"type":"message","role":"user","content":[
				  {"type":"input_text","text":"Describe"},
				  {"type":"input_image","image_url":"data:image/png;base64,AAA","detail":"auto"},
				  {"type":"input_file","filename":"report.pdf","file_data":"data:application/pdf;base64,BBB"}]}]
				""");
	}

	@Test
	void skipsEmptyAssistantTextAndDefaultsBlankToolArguments() {
		ObjectNode body = map(
				"""
						{"model":"gpt-test","stream":true,"messages":[
						  {"role":"user","content":"Time?"},
						  {"role":"assistant","content":"","tool_calls":[{"id":"call_1","type":"function","function":{"name":"time","arguments":""}}]}]}
						""");

		assertJson(body.path("input"), """
				[{"type":"message","role":"user","content":"Time?"},
				 {"type":"function_call","call_id":"call_1","name":"time","arguments":"{}"}]
				""");
	}

	@ParameterizedTest
	@ValueSource(strings = { "\"stop\":[\"END\"]", "\"n\":2", "\"seed\":7", "\"frequency_penalty\":0.5",
			"\"presence_penalty\":0.5", "\"logit_bias\":{\"50256\":-100}", "\"logprobs\":true", "\"top_logprobs\":2",
			"\"modalities\":[\"audio\"]", "\"web_search_options\":{\"search_context_size\":\"low\"}" })
	void rejectsChatParametersWithoutResponsesEquivalent(String parameter) {
		String request = "{\"model\":\"gpt-test\",\"stream\":true,\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"}],"
				+ parameter + "}";

		assertThatThrownBy(() -> map(request)).isInstanceOf(IllegalArgumentException.class)
			.hasMessageContaining("not supported by the OpenAI Responses API");
	}

	@Test
	void rejectsAudioInput() {
		assertThatThrownBy(() -> map("""
				{"model":"gpt-test","stream":true,"messages":[{"role":"user","content":[
				  {"type":"input_audio","input_audio":{"data":"AAA","format":"wav"}}]}]}
				""")).isInstanceOf(IllegalArgumentException.class).hasMessageContaining("Audio input");
	}

}
