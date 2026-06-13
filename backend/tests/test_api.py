"""API behavior tests (fake pipeline + in-memory DB via conftest)."""


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_observe_applies_and_routes_to_review(client):
    resp = client.post(
        "/inference/observe",
        files={"image": ("rack.png", b"fake-bytes", "image/png")},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["observations"] == 2
    assert data["applied"] == 1
    assert data["review"] == 1


def test_inventory_reflects_applied_change(client):
    client.post(
        "/inference/observe",
        files={"image": ("rack.png", b"fake-bytes", "image/png")},
    )

    products = client.get("/inventory/products").json()
    assert products == [{"sku": "SKU-1", "name": "widget", "quantity": 1}]

    review = client.get("/inventory/review").json()
    assert len(review) == 1
    assert review[0]["disposition"] == "review"


def test_two_observations_accumulate_quantity(client):
    for _ in range(2):
        client.post(
            "/inference/observe",
            files={"image": ("rack.png", b"fake-bytes", "image/png")},
        )
    products = client.get("/inventory/products").json()
    assert products[0]["quantity"] == 2
