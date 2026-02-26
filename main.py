#!/usr/bin/env python3
import os
from flask import Flask, send_from_directory, jsonify


# Application factory moved here from app/__init__.py
# static_folder needs to point at the package's static directory.

def create_app():
    static_dir = os.path.join(os.path.dirname(__file__), "app", "static")
    server = Flask(__name__, static_folder=static_dir, static_url_path="/static")

    @server.route("/")
    def index():
        return send_from_directory(server.static_folder, "index.html")

    @server.route("/assets/<path:filename>")
    def assets(filename):
        return send_from_directory(os.path.join(server.static_folder, "assets"), filename)

    # mount dash apps lazily
    from app.dash_apps.farfield import create_farfield_dash
    from app.dash_apps.farfield_range_beams import create_farfield_range_beams_dash
    from app.dash_apps.nearfield_range_beams import create_nearfield_range_beams_dash
    from app.dash_apps.nearfield import create_nearfield_dash
    from app.dash_apps.farfield_range_of_beams_sub6 import create_farfield_range_of_beams_sub6_dash

    create_farfield_dash(server, url_base_pathname="/dash/far_field/")
    create_farfield_range_beams_dash(server, url_base_pathname="/dash/far_field_range_of_beams/")
    create_nearfield_dash(server, url_base_pathname="/dash/near_field/")
    create_nearfield_range_beams_dash(server, url_base_pathname="/dash/near_field_range_of_beams/")
    create_farfield_range_of_beams_sub6_dash(server, url_base_pathname="/dash/far_field_range_of_beams_sub6/")

    @server.route("/api/health")
    def health():
        return jsonify({"status": "ok"})

    return server


def run():
    app = create_app()

    debug_flag = os.environ.get("DEBUG", os.environ.get("FLASK_DEBUG", "False")).lower() in ("1", "true", "yes")
    port = int(os.environ.get("PORT", 8050))
    app.run(host="0.0.0.0", port=port, debug=debug_flag)


if __name__ == "__main__":
    run()
