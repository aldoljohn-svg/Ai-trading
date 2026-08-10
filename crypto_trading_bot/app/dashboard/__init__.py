"""Live web dashboard.

Route handlers are pure functions of the engine, so the same routes are served
by FastAPI when it is installed and by a small ``http.server`` fallback when it
is not.  The fallback exists so ``/health`` is always reachable - a monitoring
endpoint that disappears when a dependency is missing is not a monitoring
endpoint.
"""

from app.dashboard.api import DashboardRoutes, create_app
from app.dashboard.server import FallbackDashboardServer, serve_dashboard

__all__ = [
    "DashboardRoutes",
    "create_app",
    "FallbackDashboardServer",
    "serve_dashboard",
]
