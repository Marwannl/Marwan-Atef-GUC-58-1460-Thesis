from routers.chat import ChatRequest, _build_user_content


def test_chat_request_accepts_image():
    req = ChatRequest(chat_id=1, message="hello", image_data_url="data:image/png;base64,abc")
    assert req.image_data_url == "data:image/png;base64,abc"


def test_chat_request_image_optional():
    req = ChatRequest(chat_id=1, message="hello")
    assert req.image_data_url is None


def test_build_user_content_text_only():
    result = _build_user_content("hello", None)
    assert result == "hello"


def test_build_user_content_with_image():
    result = _build_user_content("look at this", "data:image/png;base64,abc")
    assert result == [
        {"type": "text", "text": "look at this"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
    ]
