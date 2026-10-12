"""Wildlife Insights login and signed image URL resolution."""

import json
import re
import threading

import requests

from src.utils.credentials import get_credential

_wi_token = None
_wi_token_lock = threading.Lock()


def get_wildlife_insights_token(cookie=None, refresh=False):
    """Fetches or refreshes a Wildlife Insights JWT token using the session cookie or environment variable."""
    global _wi_token
    with _wi_token_lock:
        if _wi_token and not refresh:
            return _wi_token
        token = get_credential("WILDLIFE_INSIGHTS_TOKEN")
        if token and not refresh:
            _wi_token = token
            return _wi_token
        cookie = cookie or get_credential("WILDLIFE_INSIGHTS_COOKIE")
        if not cookie:
            return None
        try:
            r = requests.get(
                "https://app.wildlifeinsights.org/explore",
                headers={"Cookie": cookie, "User-Agent": "Mozilla/5.0"},
                timeout=10,
            )
            match = re.search(
                r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', r.text
            ) or re.search(r"__NEXT_DATA__\s*=\s*(.*?);__NEXT_LOADED_PAGES__", r.text)
            if match:
                data = json.loads(match.group(1))
                token_val = (
                    data.get("props", {})
                    .get("initialState", {})
                    .get("auth", {})
                    .get("token")
                )
                if token_val:
                    _wi_token = token_val
                    return _wi_token
        except Exception:
            pass
        return None


def resolve_wildlife_insights_url(url, cookie=None, token=None, photo_id=None):
    """Resolves a Wildlife Insights public web viewer URL into a signed Google Cloud Storage image URL."""
    if not url or "wildlifeinsights.org" not in url:
        return None

    m_dl = re.search(r"/download/(\d+)", url)
    m_proj = re.search(r"/project/(\d+)", url)
    m_img = re.search(r"/data-files/([a-zA-Z0-9\-]+)", url)

    dl_id = int(m_dl.group(1)) if m_dl else None
    proj_id = int(m_proj.group(1)) if m_proj else None
    img_uuid = m_img.group(1) if m_img else str(photo_id or "")

    if not dl_id or not img_uuid:
        return None

    auth_token = token or get_wildlife_insights_token(cookie)
    if not auth_token:
        return None

    gql_ep = "https://api.wildlifeinsights.org/graphql"
    query = """query getDataFilePublicDownloadUrl($downloadId: Int!, $projectId: Int, $imageUUID: String!) {
  getDataFilePublicDownloadUrl(downloadId: $downloadId, projectId: $projectId, imageUUID: $imageUUID) {
    url
  }
}"""
    variables = {
        "downloadId": dl_id,
        "projectId": proj_id,
        "imageUUID": img_uuid,
    }
    headers = {
        "Authorization": f"Bearer {auth_token}",
        "Content-Type": "application/json",
        "User-Agent": "Mozilla/5.0",
    }

    try:
        r = requests.post(
            gql_ep,
            json={"query": query, "variables": variables},
            headers=headers,
            timeout=10,
        )
        if r.status_code == 401:
            auth_token = get_wildlife_insights_token(cookie, refresh=True)
            if auth_token:
                headers["Authorization"] = f"Bearer {auth_token}"
                r = requests.post(
                    gql_ep,
                    json={"query": query, "variables": variables},
                    headers=headers,
                    timeout=10,
                )
        if r.status_code == 200:
            res_data = r.json()
            if (
                "data" in res_data
                and res_data["data"]
                and res_data["data"].get("getDataFilePublicDownloadUrl")
            ):
                return res_data["data"]["getDataFilePublicDownloadUrl"].get("url")
    except Exception:
        pass
    return None
