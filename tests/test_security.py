import time

import jwt
import pytest
from bson import ObjectId

from app.config import settings
from app.core.exceptions import AuthenticationError
from app.core.security import decode_access_token
from app.models.authModel import CurrentUserMapper

SECRET = settings.SECRET_KEY.get_secret_value()


def _token(payload: dict, secret: str = SECRET) -> str:
    return jwt.encode(payload, secret, algorithm="HS256")


def _payload(**overrides: object) -> dict:
    payload = {
        "user_id": str(ObjectId()),
        "org_id": str(ObjectId()),
        "permissions": ["NGAUTOMATE:ACCESS:OWN_ORG"],
        "exp": int(time.time()) + 60,
    }
    payload.update(overrides)
    return payload


def test_valid_token_is_decoded_into_a_user() -> None:
    payload = _payload()

    user = CurrentUserMapper.from_token_payload(decode_access_token(_token(payload)))

    assert user.user_id == payload["user_id"]
    assert user.org_id == payload["org_id"]
    assert user.has_permission("NGAUTOMATE:ACCESS:OWN_ORG")


@pytest.mark.parametrize(
    "token",
    [
        pytest.param(_token(_payload(exp=int(time.time()) - 10)), id="expired"),
        pytest.param(_token(_payload(), secret="another-secret-that-is-also-32-bytes-long"), id="wrong-secret"),
        pytest.param(_token({k: v for k, v in _payload().items() if k != "exp"}), id="no-expiry"),
        pytest.param("not-a-jwt", id="garbage"),
    ],
)
def test_bad_tokens_are_rejected(token: str) -> None:
    with pytest.raises(AuthenticationError):
        decode_access_token(token)


@pytest.mark.parametrize("field", ["user_id", "org_id"])
def test_token_without_valid_ids_is_rejected(field: str) -> None:
    with pytest.raises(AuthenticationError):
        CurrentUserMapper.from_token_payload(_payload(**{field: "not-an-object-id"}))


def test_permissions_that_are_not_a_list_count_as_none() -> None:
    user = CurrentUserMapper.from_token_payload(_payload(permissions="NGAUTOMATE:ACCESS:OWN_ORG"))

    assert user.permissions == frozenset()
