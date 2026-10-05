from pathlib import Path
from fastapi.testclient import TestClient
from telemachus.service import create_app

ROOT=Path(__file__).resolve().parents[1]


def test_service_schema_and_real_execution():
    client=TestClient(create_app(ROOT))
    assert client.get("/health").json()["version"]=="5.0.0"
    assert client.post("/perceive",json={"tiles":[[1,2]]}).status_code==422
    assert client.post("/episode",json={"method":"oracle"}).status_code==422
    assert client.post("/ground-mission",json={"image_base64":"not_an_image"}).status_code==422
    response=client.post("/episode",json={"seed":100,"max_steps":3})
    assert response.status_code==200
    result=response.json()
    assert result["steps"]<=3 and result["resources"]["cost"]<=120
