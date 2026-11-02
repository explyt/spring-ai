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

import okhttp3.mockwebserver.MockResponse;
import okhttp3.mockwebserver.RecordedRequest;
import tools.jackson.databind.JsonNode;

import org.springframework.ai.model.ModelOptionsUtils;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;

public final class ResponsesApiFixtures {

	public static final String CREATED = """
			{"type":"response.created","response":{"id":"resp_1","created_at":1700000000,"model":"gpt-test","status":"in_progress","output":[]}}""";

	public static final String TEXT_HEL = """
			{"type":"response.output_text.delta","item_id":"msg_1","output_index":0,"content_index":0,"delta":"Hel"}""";

	public static final String TEXT_LO = """
			{"type":"response.output_text.delta","item_id":"msg_1","output_index":0,"content_index":0,"delta":"lo"}""";

	public static final String TEXT_SUNNY = """
			{"type":"response.output_text.delta","item_id":"msg_2","output_index":0,"content_index":0,"delta":"Sunny"}""";

	public static final String CALL_PARIS_DONE = """
			{"type":"response.output_item.done","output_index":0,"item":{"id":"fc_1","type":"function_call","status":"completed","call_id":"call_1","name":"weather","arguments":"{\\"city\\":\\"Paris\\"}"}}""";

	public static final String CALL_BERLIN_DONE = """
			{"type":"response.output_item.done","output_index":1,"item":{"id":"fc_2","type":"function_call","status":"completed","call_id":"call_2","name":"weather","arguments":"{\\"city\\":\\"Berlin\\"}"}}""";

	public static final String COMPLETED = """
			{"type":"response.completed","response":{"id":"resp_1","created_at":1700000000,"model":"gpt-test","status":"completed","output":[],"usage":{"input_tokens":9,"output_tokens":12,"total_tokens":21}}}""";

	public static final String INCOMPLETE_MAX_OUTPUT_TOKENS = """
			{"type":"response.incomplete","response":{"id":"resp_1","created_at":1700000000,"model":"gpt-test","status":"incomplete","incomplete_details":{"reason":"max_output_tokens"},"output":[],"usage":{"input_tokens":1,"output_tokens":2,"total_tokens":3}}}""";

	public static final String FAILED = """
			{"type":"response.failed","response":{"id":"resp_1","status":"failed","error":{"code":"server_error","message":"boom"}}}""";

	public static final String RESPONSE_OBJECT = """
			{"id":"resp_1","object":"response","created_at":1700000000,"model":"gpt-test","status":"completed","output":[{"type":"reasoning","id":"rs_1","summary":[{"type":"summary_text","text":"Plan"}]},{"type":"message","id":"msg_1","role":"assistant","status":"completed","content":[{"type":"output_text","text":"Hello","annotations":[]}]}],"usage":{"input_tokens":9,"output_tokens":12,"total_tokens":21}}""";

	public static final String CHAT_COMPLETION = """
			{"id":"chatcmpl-1","object":"chat.completion","created":1700000000,"model":"gpt-test","choices":[{"index":0,"message":{"role":"assistant","content":"Hello"},"finish_reason":"stop"}],"usage":{"prompt_tokens":9,"completion_tokens":12,"total_tokens":21}}""";

	private ResponsesApiFixtures() {
	}

	public static MockResponse sse(String... dataFrames) {
		StringBuilder body = new StringBuilder();
		for (String data : dataFrames) {
			String type = ModelOptionsUtils.JSON_MAPPER.readTree(data).path("type").asText();
			body.append("event: ").append(type).append('\n').append("data: ").append(data).append("\n\n");
		}
		return new MockResponse().setResponseCode(200)
			.addHeader(HttpHeaders.CONTENT_TYPE, MediaType.TEXT_EVENT_STREAM_VALUE)
			.setBody(body.toString());
	}

	public static MockResponse json(String body) {
		return new MockResponse().setResponseCode(200)
			.addHeader(HttpHeaders.CONTENT_TYPE, MediaType.APPLICATION_JSON_VALUE)
			.setBody(body);
	}

	public static JsonNode body(RecordedRequest request) {
		return ModelOptionsUtils.JSON_MAPPER.readTree(request.getBody().readUtf8());
	}

}
