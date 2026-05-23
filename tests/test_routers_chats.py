def _register_and_login(client, username="testuser", password="pass1234"):
    client.post("/auth/register", json={"username": username, "password": password})
    resp = client.post("/auth/login", json={"username": username, "password": password})
    return {"Authorization": f"Bearer {resp.json()['access_token']}"}


def test_create_chat(client):
    headers = _register_and_login(client)
    response = client.post("/chats", headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert data["title"] == "New Chat"
    assert "id" in data


def test_list_chats(client):
    headers = _register_and_login(client)
    client.post("/chats", headers=headers)
    client.post("/chats", headers=headers)
    response = client.get("/chats", headers=headers)
    assert response.status_code == 200
    assert len(response.json()) == 2


def test_delete_chat(client):
    headers = _register_and_login(client)
    chat_id = client.post("/chats", headers=headers).json()["id"]
    response = client.delete(f"/chats/{chat_id}", headers=headers)
    assert response.status_code == 200
    assert client.get("/chats", headers=headers).json() == []


def test_cannot_access_other_users_chat(client):
    headers1 = _register_and_login(client, "user1", "pass1234")
    headers2 = _register_and_login(client, "user2", "pass5678")
    chat_id = client.post("/chats", headers=headers1).json()["id"]
    response = client.get(f"/chats/{chat_id}/messages", headers=headers2)
    assert response.status_code == 403


def test_unauthenticated_access(client):
    response = client.get("/chats")
    assert response.status_code == 403
