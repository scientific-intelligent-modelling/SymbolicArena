from __future__ import annotations

import argparse
import hashlib
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from configparser import ConfigParser
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

import oss2
from oss2.credentials import StaticCredentialsProvider
from oss2.exceptions import NoSuchKey
from oss2.models import BucketCors, BucketCreateConfig, CorsRule


REGION = "cn-hongkong"
ENDPOINT = "https://oss-cn-hongkong.aliyuncs.com"
SOURCE_BUCKET = "symbolicarena-data"
PUBLIC_ORIGINS = ("https://symbolicarena.top", "http://symbolicarena.top")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def credentials() -> oss2.ProviderAuthV4:
    path = Path.home() / ".config/aliyun/oss-credentials.ini"
    settings = ConfigParser()
    if settings.read(path) != [str(path)]:
        raise FileNotFoundError(path)
    profile = settings["default"]
    provider = StaticCredentialsProvider(profile["access_key_id"], profile["access_key_secret"])
    return oss2.ProviderAuthV4(provider)


def release_files(release_root: Path) -> list[dict[str, Any]]:
    manifest_path = release_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["release"] != release_root.name:
        raise ValueError("发布名称与清单不一致")
    files = list(manifest["files"])
    files.append({
        "path": "manifest.json",
        "bytes": manifest_path.stat().st_size,
        "sha256": sha256_file(manifest_path),
    })
    paths = [item["path"] for item in files]
    if len(paths) != len(set(paths)):
        raise ValueError("发布清单中存在重复路径")
    for item in files:
        file_path = (release_root / item["path"]).resolve()
        if not file_path.is_relative_to(release_root) or not file_path.is_file():
            raise ValueError(f"发布文件不存在或路径越界：{item['path']}")
        if file_path.stat().st_size != item["bytes"] or sha256_file(file_path) != item["sha256"]:
            raise ValueError(f"发布文件与清单不一致：{item['path']}")
    return files


def get_bucket(auth: oss2.ProviderAuthV4) -> tuple[oss2.Bucket, str]:
    private_bucket = oss2.Bucket(auth, ENDPOINT, SOURCE_BUCKET, region=REGION)
    owner = private_bucket.get_bucket_info().owner.id
    if not owner or not owner.isdecimal():
        raise ValueError("无法确认 OSS 账号 ID")
    name = f"symbolicarena-pages-{owner}"
    return oss2.Bucket(auth, ENDPOINT, name, region=REGION), owner


def ensure_bucket(bucket: oss2.Bucket, owner: str, auth: oss2.ProviderAuthV4) -> None:
    service = oss2.Service(auth, ENDPOINT, region=REGION)
    names = {item.name for item in oss2.BucketIterator(service)}
    if bucket.bucket_name not in names:
        config = BucketCreateConfig(
            oss2.BUCKET_STORAGE_CLASS_STANDARD,
            oss2.BUCKET_DATA_REDUNDANCY_TYPE_LRS,
        )
        bucket.create_bucket(permission=oss2.BUCKET_ACL_PRIVATE, input=config)
    info = bucket.get_bucket_info()
    if info.owner.id != owner or info.location != f"oss-{REGION}" or info.acl.grant != "private":
        raise ValueError("发布专用存储空间的所有者、地域或访问权限不符合要求")


def object_key(release: str, relative_path: str) -> str:
    return f"web/releases/{release}/{relative_path}"


def head_matches(bucket: oss2.Bucket, key: str, item: dict[str, Any]) -> bool:
    try:
        response = bucket.head_object(key)
    except NoSuchKey:
        return False
    remote_hash = response.headers.get("x-oss-meta-sha256")
    if int(response.headers["Content-Length"]) != item["bytes"] or remote_hash != item["sha256"]:
        raise ValueError(f"OSS 已存在不同内容的对象：{key}")
    return True


def upload_files(auth: oss2.ProviderAuthV4, name: str, release_root: Path, files: list[dict[str, Any]], workers: int) -> None:
    local = threading.local()

    def upload(item: dict[str, Any]) -> str:
        if not hasattr(local, "bucket"):
            local.bucket = oss2.Bucket(auth, ENDPOINT, name, region=REGION)
        bucket = local.bucket
        key = object_key(release_root.name, item["path"])
        if head_matches(bucket, key, item):
            return "existing"
        headers = {
            "Content-Type": "application/gzip" if item["path"].endswith(".gz") else "application/json; charset=utf-8",
            "Cache-Control": "public,max-age=31536000,immutable",
            "x-oss-meta-sha256": item["sha256"],
        }
        result = bucket.put_object_from_file(key, str(release_root / item["path"]), headers=headers)
        if result.status != 200 or not head_matches(bucket, key, item):
            raise ValueError(f"OSS 上传后核验失败：{key}")
        return "uploaded"

    counts = {"existing": 0, "uploaded": 0}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(upload, item) for item in files]
        for future in as_completed(futures):
            counts[future.result()] += 1
            completed = counts["existing"] + counts["uploaded"]
            if completed % 250 == 0 or completed == len(files):
                print(f"OSS 对象已核验 {completed}/{len(files)}", flush=True)
    print("新增对象", counts["uploaded"], "已有对象", counts["existing"], flush=True)


def configure_public_read(bucket: oss2.Bucket, owner: str) -> None:
    rule = CorsRule(
        allowed_origins=list(PUBLIC_ORIGINS),
        allowed_methods=["GET", "HEAD"],
        allowed_headers=[],
        expose_headers=["ETag", "Content-Length"],
        max_age_seconds=86400,
    )
    bucket.put_bucket_cors(BucketCors(rules=[rule], response_vary=True))
    bucket.put_bucket_public_access_block(False)
    policy = {
        "Version": "1",
        "Statement": [{
            "Effect": "Allow",
            "Action": ["oss:GetObject"],
            "Principal": ["*"],
            "Resource": [f"acs:oss:*:{owner}:{bucket.bucket_name}/web/*"],
        }],
    }
    bucket.put_bucket_policy(json.dumps(policy, separators=(",", ":")))


def verify_public_catalog(name: str, release_root: Path) -> str:
    key = object_key(release_root.name, "catalog.json")
    url = f"https://{name}.oss-cn-hongkong.aliyuncs.com/{key}"
    for origin in PUBLIC_ORIGINS:
        request = Request(url, headers={"Origin": origin})
        with urlopen(request, timeout=30) as response:
            if response.status != 200 or response.headers.get("Access-Control-Allow-Origin") != origin:
                raise ValueError("公开目录文件或跨域响应核验失败")
            if hashlib.sha256(response.read()).hexdigest() != sha256_file(release_root / "catalog.json"):
                raise ValueError("公开目录文件内容与本地不一致")
    return url


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--configure-only", action="store_true")
    args = parser.parse_args()
    release_root = args.release_root.resolve()
    files = release_files(release_root)
    auth = credentials()
    bucket, owner = get_bucket(auth)
    print("发布专用存储空间", bucket.bucket_name, "文件数量", len(files), flush=True)
    if not args.publish:
        return
    ensure_bucket(bucket, owner, auth)
    if not args.configure_only:
        upload_files(auth, bucket.bucket_name, release_root, files, args.workers)
    configure_public_read(bucket, owner)
    print("公开目录地址", verify_public_catalog(bucket.bucket_name, release_root), flush=True)


if __name__ == "__main__":
    main()
