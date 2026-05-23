def test_register_new_user(client):
    response = client.post("/auth/register", json={"username": "alice", "password": "pass1234"})
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"


def test_register_duplicate_username(client):
    client.post("/auth/register", json={"username": "alice", "password": "pass1234"})
    response = client.post("/auth/register", json={"username": "alice", "password": "other"})
    assert response.status_code == 400


def test_login_valid_credentials(client):
    client.post("/auth/register", json={"username": "bob", "password": "pass1234"})
    response = client.post("/auth/login", json={"username": "bob", "password": "pass1234"})
    assert response.status_code == 200
    assert "access_token" in response.json()


def test_login_wrong_password(client):
    client.post("/auth/register", json={"username": "carol", "password": "pass1234"})
    response = client.post("/auth/login", json={"username": "carol", "password": "wrong"})
    assert response.status_code == 401
