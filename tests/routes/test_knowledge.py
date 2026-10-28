"""Route tests for ``/v3/workspaces/{workspace_id}/knowledge``.

Covers the frozen HTTP contract (status codes, payload shapes, validation and
auth) against the isolated test database and the deterministic autouse embedding
mock -- no real Honcho server, no real embedding API.
"""

import hashlib

import pytest
from fastapi.testclient import TestClient

from src.config import settings
from src.embedding_client import EmbeddingTokenLimitError
from src.security import JWTParams, create_jwt


def _url(workspace_name: str, suffix: str = "") -> str:
    return f"/v3/workspaces/{workspace_name}/knowledge{suffix}"


def _payload(**overrides: object) -> dict[str, object]:
    body: dict[str, object] = {
        "title": "HNSW 索引参数",
        "content": "m=16, ef_construction=64 是现有 documents 表的取值。",
        "source": "honcho/src/models.py",
        "tags": ["honcho", "pgvector"],
    }
    body.update(overrides)
    return body


def _create(client: TestClient, workspace_name: str, **overrides: object):
    return client.post(_url(workspace_name), json=_payload(**overrides))


def _sha256(text: str) -> str:
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


class TestCreateRoute:
    """POST /knowledge -- save, idempotency, validation, failure mapping."""

    def test_creates_entry_and_returns_201(self, client, sample_data):
        workspace, _ = sample_data

        response = _create(client, workspace.name)

        assert response.status_code == 201
        body = response.json()
        assert body["created"] is True
        entry = body["entry"]
        assert entry["title"] == "HNSW 索引参数"
        assert entry["source"] == "honcho/src/models.py"
        assert entry["tags"] == ["honcho", "pgvector"]
        assert entry["content_hash"] == _sha256(
            "m=16, ef_construction=64 是现有 documents 表的取值。"
        )
        assert len(entry["id"]) == 21
        assert entry["created_at"] and entry["updated_at"]
        assert "embedding" not in entry

    def test_duplicate_returns_200_and_embeds_once(
        self, client, sample_data, mock_openai_embeddings
    ):
        workspace, _ = sample_data

        first = _create(client, workspace.name)
        second = _create(client, workspace.name, title="完全不同的标题")

        assert first.status_code == 201
        assert second.status_code == 200
        assert second.json()["created"] is False
        assert second.json()["entry"]["id"] == first.json()["entry"]["id"]
        # Duplicate metadata is NOT rewritten by a re-save.
        assert second.json()["entry"]["title"] == "HNSW 索引参数"
        mock_openai_embeddings["embed"].assert_awaited_once()

    def test_ignores_client_supplied_id_hash_and_timestamps(self, client, sample_data):
        workspace, _ = sample_data

        response = _create(
            client,
            workspace.name,
            id="clientSuppliedId00000",
            content_hash="deadbeef",
            created_at="2020-01-01T00:00:00Z",
            updated_at="2020-01-01T00:00:00Z",
        )

        assert response.status_code in (200, 201)
        entry = response.json()["entry"]
        assert entry["id"] != "clientSuppliedId00000"
        assert entry["content_hash"] != "deadbeef"
        assert not entry["created_at"].startswith("2020-01-01")

    def test_normalizes_whitespace_and_dedupes_tags(self, client, sample_data):
        workspace, _ = sample_data

        response = _create(
            client,
            workspace.name,
            title="  spaced title  ",
            content="  spaced body  ",
            source="  docs/source.md  ",
            tags=["  a  ", "a", " b "],
        )

        assert response.status_code == 201
        entry = response.json()["entry"]
        assert entry["title"] == "spaced title"
        assert entry["content"] == "spaced body"
        assert entry["source"] == "docs/source.md"
        assert entry["tags"] == ["a", "b"]

    @pytest.mark.parametrize(
        "overrides",
        [
            {"source": "   "},
            {"title": ""},
            {"content": ""},
            {"title": "x" * 201},
            {"content": "x" * 4001},
            {"source": "x" * 501},
            {"tags": [f"tag{index}" for index in range(11)]},
            {"tags": ["ok", "  "]},
            {"tags": ["x" * 33]},
            {"tags": "not-a-list"},
            {"content": 42},
        ],
    )
    def test_rejects_invalid_payload(self, client, sample_data, overrides):
        workspace, _ = sample_data

        response = _create(client, workspace.name, **overrides)

        assert response.status_code == 422

    def test_missing_required_field_is_422(self, client, sample_data):
        workspace, _ = sample_data

        response = client.post(
            _url(workspace.name),
            json={"title": "t", "content": "c"},
        )

        assert response.status_code == 422

    def test_unknown_workspace_is_404(self, client, sample_data):
        response = _create(client, "no-such-workspace")

        assert response.status_code == 404
        assert response.json()["detail"] == "Workspace no-such-workspace not found"

    def test_embedding_unavailable_is_503(
        self, client, sample_data, mock_openai_embeddings
    ):
        workspace, _ = sample_data
        mock_openai_embeddings["embed"].side_effect = RuntimeError("connection reset")

        response = _create(client, workspace.name)

        assert response.status_code == 503
        assert response.json()["detail"] == "embedding service unavailable"

    def test_token_limit_is_422(self, client, sample_data, mock_openai_embeddings):
        workspace, _ = sample_data
        mock_openai_embeddings["embed"].side_effect = EmbeddingTokenLimitError(
            "too long"
        )

        response = _create(client, workspace.name)

        assert response.status_code == 422
        assert "exceeds maximum token limit" in response.json()["detail"]


class TestReadRoutes:
    """GET /knowledge and GET /knowledge/{entry_id}."""

    def test_get_returns_entry(self, client, sample_data):
        workspace, _ = sample_data
        entry_id = _create(client, workspace.name).json()["entry"]["id"]

        response = client.get(_url(workspace.name, f"/{entry_id}"))

        assert response.status_code == 200
        assert response.json()["entry"]["id"] == entry_id

    def test_get_missing_entry_is_404(self, client, sample_data):
        workspace, _ = sample_data

        response = client.get(_url(workspace.name, "/missing-id"))

        assert response.status_code == 404
        assert response.json()["detail"] == "knowledge entry missing-id not found"

    async def test_get_other_workspace_entry_is_404(
        self, client, db_session, sample_data
    ):
        from nanoid import generate as generate_nanoid

        from src import models

        workspace, _ = sample_data
        other = models.Workspace(name=str(generate_nanoid()))
        db_session.add(other)
        await db_session.commit()
        entry_id = _create(client, workspace.name).json()["entry"]["id"]

        response = client.get(_url(other.name, f"/{entry_id}"))

        assert response.status_code == 404

    def test_list_returns_page_with_total(self, client, sample_data):
        workspace, _ = sample_data
        ids = [
            _create(client, workspace.name, content=f"body {index}").json()["entry"][
                "id"
            ]
            for index in range(3)
        ]

        response = client.get(_url(workspace.name), params={"limit": 2, "offset": 0})

        assert response.status_code == 200
        body = response.json()
        assert body["total"] == 3
        assert body["limit"] == 2
        assert body["offset"] == 0
        assert len(body["items"]) == 2
        assert {item["id"] for item in body["items"]} <= set(ids)

    def test_list_defaults(self, client, sample_data):
        workspace, _ = sample_data
        _create(client, workspace.name)

        response = client.get(_url(workspace.name))

        assert response.status_code == 200
        body = response.json()
        assert (body["limit"], body["offset"], body["total"]) == (20, 0, 1)

    @pytest.mark.parametrize(
        "params",
        [{"limit": 0}, {"limit": 101}, {"offset": -1}, {"limit": "abc"}],
    )
    def test_list_rejects_bad_paging(self, client, sample_data, params):
        workspace, _ = sample_data

        response = client.get(_url(workspace.name), params=params)

        assert response.status_code == 422

    def test_list_unknown_workspace_is_404(self, client, sample_data):
        response = client.get(_url("no-such-workspace"))

        assert response.status_code == 404


class TestQueryRoute:
    """POST /knowledge/query -- hits carry source and real distance."""

    def test_query_returns_hits_with_distance(self, client, sample_data):
        workspace, _ = sample_data
        payload = _payload()
        entry_id = _create(client, workspace.name).json()["entry"]["id"]

        # Same text the save path embedded -> cosine distance 0.0.
        query = f"{payload['title']}\n{payload['content']}"
        response = client.post(
            _url(workspace.name, "/query"), json={"query": query, "top_k": 5}
        )

        assert response.status_code == 200
        body = response.json()
        assert body["query"] == query
        assert len(body["hits"]) == 1
        hit = body["hits"][0]
        assert hit["entry"]["id"] == entry_id
        assert hit["entry"]["source"] == "honcho/src/models.py"
        assert hit["distance"] == pytest.approx(0.0, abs=1e-6)

    def test_no_match_is_200_with_empty_hits(self, client, sample_data):
        workspace, _ = sample_data
        _create(client, workspace.name)

        response = client.post(
            _url(workspace.name, "/query"),
            json={"query": "完全不相关的问题", "top_k": 5, "max_distance": 0.0},
        )

        assert response.status_code == 200
        assert response.json()["hits"] == []

    def test_empty_workspace_query_is_200(self, client, sample_data):
        workspace, _ = sample_data

        response = client.post(
            _url(workspace.name, "/query"), json={"query": "anything"}
        )

        assert response.status_code == 200
        assert response.json()["hits"] == []

    @pytest.mark.parametrize(
        "body",
        [
            {"query": "   "},
            {"query": "x" * 1001},
            {"query": "ok", "top_k": 0},
            {"query": "ok", "top_k": 51},
            {"query": "ok", "max_distance": 2.5},
            {},
        ],
    )
    def test_query_rejects_invalid_body(self, client, sample_data, body):
        workspace, _ = sample_data

        response = client.post(_url(workspace.name, "/query"), json=body)

        assert response.status_code == 422

    def test_query_unknown_workspace_is_404(self, client, sample_data):
        response = client.post(
            _url("no-such-workspace", "/query"), json={"query": "anything"}
        )

        assert response.status_code == 404

    def test_query_embedding_unavailable_is_503(
        self, client, sample_data, mock_openai_embeddings
    ):
        workspace, _ = sample_data
        mock_openai_embeddings["embed"].side_effect = RuntimeError("connection reset")

        response = client.post(
            _url(workspace.name, "/query"), json={"query": "anything"}
        )

        assert response.status_code == 503
        assert response.json()["detail"] == "embedding service unavailable"


class TestUpdateRoute:
    """PATCH /knowledge/{entry_id} -- partial update and reembed signalling."""

    def test_patch_title_reembeds(self, client, sample_data):
        workspace, _ = sample_data
        entry_id = _create(client, workspace.name).json()["entry"]["id"]

        response = client.patch(
            _url(workspace.name, f"/{entry_id}"), json={"title": "新标题"}
        )

        assert response.status_code == 200
        body = response.json()
        assert body["reembedded"] is True
        assert body["entry"]["title"] == "新标题"
        assert body["entry"]["id"] == entry_id

    def test_patch_content_updates_hash(self, client, sample_data):
        workspace, _ = sample_data
        entry_id = _create(client, workspace.name).json()["entry"]["id"]

        response = client.patch(
            _url(workspace.name, f"/{entry_id}"), json={"content": "新的内容"}
        )

        assert response.status_code == 200
        assert response.json()["reembedded"] is True
        assert response.json()["entry"]["content_hash"] == _sha256("新的内容")

    def test_patch_tags_only_does_not_reembed(
        self, client, sample_data, mock_openai_embeddings
    ):
        workspace, _ = sample_data
        entry_id = _create(client, workspace.name).json()["entry"]["id"]
        mock_openai_embeddings["embed"].reset_mock()

        response = client.patch(_url(workspace.name, f"/{entry_id}"), json={"tags": []})

        assert response.status_code == 200
        assert response.json()["reembedded"] is False
        assert response.json()["entry"]["tags"] == []
        mock_openai_embeddings["embed"].assert_not_awaited()

    def test_patch_unchanged_values_reports_no_reembed(self, client, sample_data):
        workspace, _ = sample_data
        entry = _create(client, workspace.name).json()["entry"]

        response = client.patch(
            _url(workspace.name, f"/{entry['id']}"),
            json={"title": entry["title"], "content": entry["content"]},
        )

        assert response.status_code == 200
        assert response.json()["reembedded"] is False
        assert response.json()["entry"]["updated_at"] == entry["updated_at"]

    def test_patch_empty_body_is_422(self, client, sample_data):
        workspace, _ = sample_data
        entry_id = _create(client, workspace.name).json()["entry"]["id"]

        response = client.patch(_url(workspace.name, f"/{entry_id}"), json={})

        assert response.status_code == 422
        assert response.json()["detail"] == "no fields to update"

    def test_patch_explicit_null_is_422(self, client, sample_data):
        workspace, _ = sample_data
        entry_id = _create(client, workspace.name).json()["entry"]["id"]

        response = client.patch(
            _url(workspace.name, f"/{entry_id}"), json={"title": None}
        )

        assert response.status_code == 422

    def test_patch_content_conflict_is_409(self, client, sample_data):
        workspace, _ = sample_data
        first_id = _create(client, workspace.name, content="alpha").json()["entry"][
            "id"
        ]
        second_id = _create(client, workspace.name, content="beta").json()["entry"][
            "id"
        ]

        response = client.patch(
            _url(workspace.name, f"/{first_id}"), json={"content": "beta"}
        )

        assert response.status_code == 409
        assert response.json()["detail"] == f"content already exists: {second_id}"

    def test_patch_missing_entry_is_404(self, client, sample_data):
        workspace, _ = sample_data

        response = client.patch(
            _url(workspace.name, "/missing-id"), json={"title": "x"}
        )

        assert response.status_code == 404
        assert response.json()["detail"] == "knowledge entry missing-id not found"

    def test_patch_unknown_workspace_is_404(self, client, sample_data):
        response = client.patch(
            _url("no-such-workspace", "/whatever"), json={"title": "x"}
        )

        assert response.status_code == 404


class TestDeleteRoute:
    """DELETE /knowledge/{entry_id} -- 204 then 404."""

    def test_delete_returns_204_and_is_idempotent_404_afterwards(
        self, client, sample_data
    ):
        workspace, _ = sample_data
        entry_id = _create(client, workspace.name).json()["entry"]["id"]

        deleted = client.delete(_url(workspace.name, f"/{entry_id}"))
        fetched = client.get(_url(workspace.name, f"/{entry_id}"))
        deleted_again = client.delete(_url(workspace.name, f"/{entry_id}"))

        assert deleted.status_code == 204
        assert deleted.content == b""
        assert fetched.status_code == 404
        assert deleted_again.status_code == 404

    def test_delete_missing_entry_is_404(self, client, sample_data):
        workspace, _ = sample_data

        response = client.delete(_url(workspace.name, "/missing-id"))

        assert response.status_code == 404


class TestAuth:
    """Only authenticated workspace callers reach these routes."""

    def test_missing_token_is_401(self, client, sample_data, monkeypatch):
        workspace, _ = sample_data
        monkeypatch.setattr(settings.AUTH, "USE_AUTH", True)
        monkeypatch.setattr(settings.AUTH, "JWT_SECRET", "test-secret")
        client.headers.pop("Authorization", None)

        response = client.get(_url(workspace.name))

        assert response.status_code == 401

    def test_other_workspace_token_is_rejected(self, client, sample_data, monkeypatch):
        workspace, _ = sample_data
        monkeypatch.setattr(settings.AUTH, "USE_AUTH", True)
        monkeypatch.setattr(settings.AUTH, "JWT_SECRET", "test-secret")
        token = create_jwt(JWTParams(w="some-other-workspace"))
        client.headers.pop("Authorization", None)

        response = client.get(
            _url(workspace.name), headers={"Authorization": f"Bearer {token}"}
        )

        assert response.status_code == 401

    def test_workspace_token_is_allowed(self, client, sample_data, monkeypatch):
        workspace, _ = sample_data
        monkeypatch.setattr(settings.AUTH, "USE_AUTH", True)
        monkeypatch.setattr(settings.AUTH, "JWT_SECRET", "test-secret")
        token = create_jwt(JWTParams(w=workspace.name))
        client.headers.pop("Authorization", None)

        response = client.get(
            _url(workspace.name), headers={"Authorization": f"Bearer {token}"}
        )

        assert response.status_code == 200
