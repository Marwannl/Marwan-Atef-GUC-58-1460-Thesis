from auth import hash_password, verify_password, create_access_token, decode_access_token


def test_hash_and_verify_password():
    pwd = "mysecretpassword"
    hashed = hash_password(pwd)
    assert hashed != pwd
    assert verify_password(pwd, hashed)
    assert not verify_password("wrongpassword", hashed)


def test_create_and_decode_token():
    token = create_access_token({"sub": "testuser"})
    assert isinstance(token, str)
    payload = decode_access_token(token)
    assert payload["sub"] == "testuser"


def test_decode_invalid_token():
    payload = decode_access_token("notavalidtoken")
    assert payload is None
