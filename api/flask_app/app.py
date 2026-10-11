"""Flask app (secondary API). Uses the same service layer as FastAPI. Run: python -m api.flask_app.app"""

from __future__ import annotations

import os
import uuid

from flask import Flask, jsonify, request

from api.services.trustlens_service import ModelNotLoadedError, get_service


def create_app() -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024
    p = "/api/v1"

    def run(fn):
        try:
            return jsonify(fn(request.get_json(silent=True)))
        except ModelNotLoadedError as e:
            return jsonify({"error": str(e)}), 503
        except ValueError as e:
            return jsonify({"error": str(e)}), 400

    @app.after_request
    def add_request_id(resp):
        resp.headers["X-Request-ID"] = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        return resp

    @app.get(f"{p}/health")
    def health():
        h = get_service().health()
        return jsonify(h), (200 if h["model_loaded"] else 503)

    for route, name in [
        ("predict", "predict"),
        ("batch-predict", "batch_predict"),
        ("aspect-sentiment", "aspect_sentiment"),
        ("explain", "explain"),
        ("counterfactual", "counterfactual"),
        ("similar-reviews", "similar_reviews"),
    ]:
        app.add_url_rule(
            f"{p}/{route}",
            endpoint=name,
            methods=["POST"],
            view_func=(lambda n: lambda: run(lambda body: getattr(get_service(), n)(body)))(name),
        )

    for route, name in [
        ("model-info", "model_info"),
        ("features", "features"),
        ("metrics", "metrics"),
        ("monitoring/drift", "drift"),
    ]:

        def view(n=name):
            try:
                return jsonify(getattr(get_service(), n)())
            except ModelNotLoadedError as e:
                return jsonify({"error": str(e)}), 503

        app.add_url_rule(f"{p}/{route}", endpoint=name, methods=["GET"], view_func=view)
    return app


if __name__ == "__main__":
    create_app().run(host="0.0.0.0", port=int(os.getenv("FLASK_PORT", "5000")))
