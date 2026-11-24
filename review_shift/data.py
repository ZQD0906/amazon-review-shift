"""Bounded JSONL ingestion, global audit/deduplication, disk-backed panels.

All stages process every physical nonblank JSONL line. An exact duplicate is a
canonical full-JSON duplicate; a conflicting review identity is quarantined.
No claim about the complete Amazon population follows from corpus processing.
"""
from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone, timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import time
from typing import Iterable

import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests

PIPELINE_VERSION = '1.0.0'
REVISION = '2b6d039ed471f2ba5fd2acb718bf33b0a7e5598e'
REPOSITORY = 'McAuley-Lab/Amazon-Reviews-2023'
MIN_TIMESTAMP = 820454400000  # 1996-01-01 UTC, inclusive
MAX_TIMESTAMP = 1696118400000  # 2023-10-01 UTC, exclusive
FILTER_ORDER = ['invalid_json', 'missing_id', 'invalid_rating', 'invalid_timestamp', 'accepted_for_dedup']
PANEL_COLUMNS = ['parent_asin', 'asin', 'week', 'n1', 'n2', 'n3', 'n4', 'n5', 'n', 'verified_n', 'category']
STAGE_SCHEMA = pa.schema([
    ('record_hash', pa.string()), ('identity_hash', pa.string()), ('user_hash', pa.string()),
    ('parent_asin', pa.string()), ('asin', pa.string()), ('timestamp_ms', pa.int64()),
    ('week', pa.date32()), ('rating', pa.int8()), ('verified_purchase', pa.bool_()),
    ('category', pa.string()), ('source_line', pa.int64())])

# Immutable file-byte manifest recorded from the publisher's frozen revision.
FIXED_SOURCE_SIZES = {'metadata/All_Beauty': 212990142, 'metadata/Amazon_Fashion': 1422365805, 'metadata/Appliances': 285298475, 'metadata/Arts_Crafts_and_Sewing': 2170215242, 'metadata/Automotive': 5345616548, 'metadata/Baby_Products': 691003271, 'metadata/Beauty_and_Personal_Care': 2835194976, 'metadata/Books': 14697581022, 'metadata/CDs_and_Vinyl': 948861684, 'metadata/Cell_Phones_and_Accessories': 4015159840, 'metadata/Clothing_Shoes_and_Jewelry': 17960532903, 'metadata/Digital_Music': 67097002, 'metadata/Electronics': 5246144134, 'metadata/Gift_Cards': 2038189, 'metadata/Grocery_and_Gourmet_Food': 1382105675, 'metadata/Handmade_Products': 399261431, 'metadata/Health_and_Household': 2468505069, 'metadata/Health_and_Personal_Care': 118005874, 'metadata/Home_and_Kitchen': 11788767944, 'metadata/Industrial_and_Scientific': 1130006344, 'metadata/Kindle_Store': 6865237738, 'metadata/Magazine_Subscriptions': 4096541, 'metadata/Movies_and_TV': 1285744674, 'metadata/Musical_Instruments': 631877970, 'metadata/Office_Products': 2146468843, 'metadata/Patio_Lawn_and_Garden': 2714096874, 'metadata/Pet_Supplies': 1571293235, 'metadata/Software': 256220771, 'metadata/Sports_and_Outdoors': 4132263769, 'metadata/Subscription_Boxes': 1399864, 'metadata/Tools_and_Home_Improvement': 4850689793, 'metadata/Toys_and_Games': 2637909430, 'metadata/Unknown': 674824926, 'metadata/Video_Games': 437375785, 'review/All_Beauty': 326611506, 'review/Amazon_Fashion': 1051324731, 'review/Appliances': 929451412, 'review/Arts_Crafts_and_Sewing': 3945063799, 'review/Automotive': 8727689027, 'review/Baby_Products': 2947785967, 'review/Beauty_and_Personal_Care': 11021458876, 'review/Books': 20121186727, 'review/CDs_and_Vinyl': 3287195043, 'review/Cell_Phones_and_Accessories': 9342568048, 'review/Clothing_Shoes_and_Jewelry': 27810080533, 'review/Digital_Music': 78823304, 'review/Electronics': 22616233652, 'review/Gift_Cards': 50231035, 'review/Grocery_and_Gourmet_Food': 5973991903, 'review/Handmade_Products': 289025980, 'review/Health_and_Household': 11395356745, 'review/Health_and_Personal_Care': 226990293, 'review/Home_and_Kitchen': 31408889188, 'review/Industrial_and_Scientific': 2347015351, 'review/Kindle_Store': 15800555546, 'review/Magazine_Subscriptions': 33297013, 'review/Movies_and_TV': 8391445499, 'review/Musical_Instruments': 1557845107, 'review/Office_Products': 5776792834, 'review/Patio_Lawn_and_Garden': 7748207348, 'review/Pet_Supplies': 8350775233, 'review/Software': 1869151448, 'review/Sports_and_Outdoors': 9256856041, 'review/Subscription_Boxes': 8953020, 'review/Tools_and_Home_Improvement': 12779111772, 'review/Toys_and_Games': 7316094780, 'review/Unknown': 29933791334, 'review/Video_Games': 2682948518}
FIXED_SOURCE_SHA256 = {'metadata/All_Beauty': '1cfbdf6d2ccf9c3ac52a4508b15f31f8f340508a245b65c3e3b5f461e06e97ef', 'metadata/Amazon_Fashion': '629dde620a4001934a8af705f6e5514a1b2d349720c26876ee803b90051a62df', 'metadata/Appliances': 'b25ae892af7bc452c4a54b7e4fee52f4c3e9adf058768ca3cb07ecb8d328b41e', 'metadata/Arts_Crafts_and_Sewing': 'b3dd463e900a76599494927a7e991a62660e4d5c47c3f6c4797e9dae9a65837b', 'metadata/Automotive': '8399c5ab84b2a5239e105974a33fb5d1d6d5e1e643291713008c096240ef6941', 'metadata/Baby_Products': '40250b693fc748cc0f24a5a2e762faa71e69aec4a9b934f625fd609f7cda6d7e', 'metadata/Beauty_and_Personal_Care': 'c8a69eba6579f1a3a2b6d6d564846f3c59839d0c859ff180a63a97a245c3bec2', 'metadata/Books': '3b008960580e390874b6723d438a3df3558bc98b9547ed37c90dc1659109cec7', 'metadata/CDs_and_Vinyl': 'c52275a5bce63bf293f987bff5ff1f2b268982797545f3df2b7662ab638e8aec', 'metadata/Cell_Phones_and_Accessories': 'f8f411a2e3a90560de4b18a0c2c885a4ba410bb26f965a8e7290080898d22688', 'metadata/Clothing_Shoes_and_Jewelry': 'e5ff908573938ee114e4f4dfcfc46799b0ead24a57a8c075fbca2e6e28d78382', 'metadata/Digital_Music': '7f5387e99b70631d919c87005269c555e8886ae9a60c4c59a9b6ed7881acbbec', 'metadata/Electronics': '5eb4dabd539586c49654297319ef178b62a2fe8e43108376feda125f04455d93', 'metadata/Gift_Cards': '7909903c85f0690b4ff653981724af1ea6473e7b21a7e32c7adae5026240808f', 'metadata/Grocery_and_Gourmet_Food': 'ed9b24ea707f7010b5cb321462a8e896ad1fcd59620fbc55f05084c3da59842e', 'metadata/Handmade_Products': 'c47e8f247dab7ec1d2fac93050e79a40ad5d31c3fe073838af29694fc1690530', 'metadata/Health_and_Household': '1a26906c721929a04e887eb7aac560e58e0c4f0a65ec5b24a047a560a43f0bc1', 'metadata/Health_and_Personal_Care': 'eb7f147cac85ce22e6d9b63e8f8504e85032c8a666e8634a1f101a9533983c0a', 'metadata/Home_and_Kitchen': '691353084748d985180deb13dc8de59afe4969933ad3692ea06b80460510aa56', 'metadata/Industrial_and_Scientific': '114fcde36f6205bde5d0ebed24613a35e475aa06b285011ed1e892dc26da3942', 'metadata/Kindle_Store': 'acdaba5f1fb83805a2bf032ba3c7ddb83c695b03d6c41b785a81070d3b66a145', 'metadata/Magazine_Subscriptions': 'b325bc690a4b705338dfc3dd45ace5ec643dc6212cbb0c614a9b05f236ec12ab', 'metadata/Movies_and_TV': 'beb82feef266bfa6d2e279903fd823a8b3b9c065059e1048226d0b74dc16b47d', 'metadata/Musical_Instruments': '96f40b5440918f07e3e08e84b7a6874a06a5e32a9988998ca9520cfd1b51a12c', 'metadata/Office_Products': 'd9709095587ee34b2d5e639ca69de2a6fc8ab1953c467b86715e409c5ad78314', 'metadata/Patio_Lawn_and_Garden': '3a48d1328e6d1db8728764f124f64d35b7001a73d46beeaea5d390332a5afc8e', 'metadata/Pet_Supplies': '69dfa0e6b34e8c34f5704fbf05682eb3edacac95d9b8e1fbf3f74720cbf67e50', 'metadata/Software': '7c52cce1bf3bff33f965fe117e3e8fac18a1839f88b215efa50b31a8230a5f58', 'metadata/Sports_and_Outdoors': '321aba659220c48a8dcf2a4034f1eb965fae031941d81c9b7e47811caae85db9', 'metadata/Subscription_Boxes': 'b29c7bb1e3ad1419c33142c76d39782a9b8ae2fcbe27641186b88a10e1f31575', 'metadata/Tools_and_Home_Improvement': '7b0e1ab6275a055fd1337d94dca6029eb401c31e21496608192d6564c1f25b0b', 'metadata/Toys_and_Games': 'caca4bc34518426d45e2924d1042923d9d3cd89eb69f4daeb0b3a9c87e278a9a', 'metadata/Unknown': 'be2ace4f62de51e041c838792d2fbe74a52487e8aca20f39fd729d1852e3a645', 'metadata/Video_Games': '3ad94dfaa8dfed23c5f420aa9ddec415d00f2b8ec9f2d796f487b4b2c4d256de', 'review/All_Beauty': '36ca7c48440a8247b2dca0174d095238422b506a8a80c114dca722b7715f1c94', 'review/Amazon_Fashion': 'e597b3ce6cbb2f9994a32e0eee6bb04fdae50725f2d49c89f6baef1a7b986c57', 'review/Appliances': 'd9dd03aa9be0c8e6fea66d8ce99a7e5b359e272d6323186c1aa80f6d36af1dab', 'review/Arts_Crafts_and_Sewing': 'e2f6698cdb0c1f712e38718e02efc1b74641c31bf6745be14af85583399e74df', 'review/Automotive': '5ca28ebd15a7e26f8bb3289d65e26784b39c9661d1f5dc030ecc9fef0211eced', 'review/Baby_Products': '5ab12d566ebc1162d5a197721d4da787261e036455d054ad8d858af7380e0a45', 'review/Beauty_and_Personal_Care': '24f139d51014f50c052b69bdf5ca95e2e2656e80aabda2da3fdad29bb1931a53', 'review/Books': '14f524701f3e6e850d938e09961675c397078eb7af06a99892af8fab811d5a0d', 'review/CDs_and_Vinyl': '4692a40ee4b9bcc5f7ee58edeffb211138e02641c3fdfd55f77843430cb34cbc', 'review/Cell_Phones_and_Accessories': 'bda42ca8d28a1e57db970c78703b7aaac9c9beb6595e1be00364a0a55a682cd5', 'review/Clothing_Shoes_and_Jewelry': '150eb2a9e88f61cd5a89c5337cde437dc71308c47f22899a72d9e1dad60b7356', 'review/Digital_Music': '9ac137137e55e34fce290670fdf44acee5afa2ffae541f4f3b366a835930a7c3', 'review/Electronics': 'bf11cf9c94c23ead7d3ffb174d18231fad8fcd76e72ad167f538a5599a38de8e', 'review/Gift_Cards': 'b7ceead0dd6378eebfc6f4736402f85cc08e8dcb328f813c95ca68fc62d9544e', 'review/Grocery_and_Gourmet_Food': 'bd68ea63cd101de7a0f8ae53896a0ce6b4d1d61fdef63cc02fad303e5f1244c6', 'review/Handmade_Products': '7576d7743e83806d155e506a6bce13c25d9979ab13522e9a1cf2dc593144e65c', 'review/Health_and_Household': 'e8256271e6d566745656cd946e200a43ad08ff5e046daacfc54535bfd95be4b4', 'review/Health_and_Personal_Care': 'b532849dcf8e42910ce688ad423b3c1622ce860e7bf88a3641358bb89f345d02', 'review/Home_and_Kitchen': '88f2ed8cda172b18201a9d8577ab461255b9ecec99581c1ee05c5e1b4105262b', 'review/Industrial_and_Scientific': '361f68c9535f038fab119099e79207099bee60b2cd7a98246c36b3b12c7aa678', 'review/Kindle_Store': '82024e17fdd988f47d8b5ae679643965255ae5164f9f494d6ba28ab9cc0efa76', 'review/Magazine_Subscriptions': 'bf36b04bbfb41f43c49b4adbdc13969014311ffb2344f706d13c35cdea5662da', 'review/Movies_and_TV': '4eb7428d5f48baae5a7f1132ef556e6e98181b122085ac24b34b25e033c3130a', 'review/Musical_Instruments': '63aee24d5b9eaf0069cf638703e2d62eb80fde5d2c00ca3d29881d645f6214b3', 'review/Office_Products': '82cc49670098220a314d398ceff6434599b54c27af5fa00ab9c0b4af9d9d9011', 'review/Patio_Lawn_and_Garden': '2923136d4013b1f728081ab42a488b8328164deb9b6a8e2efae8cf48ff7fe345', 'review/Pet_Supplies': 'ad9e71383eb02674a7514181bb3eb57eec36695ea2cd28143b70df5d1da72ee3', 'review/Software': 'dc119f99c466a39a09e2513c9e48eb5821fb42b1459f3026bcf71b08e6770741', 'review/Sports_and_Outdoors': '47eaab08457fd7d270613f193506c479492b696a808ae5639a949b27026a4559', 'review/Subscription_Boxes': '52e08112faa2b8769b35d364cbd3b8ac674ddd56621b171e7531a5a2071f460f', 'review/Tools_and_Home_Improvement': '7cfcd11fef2d6346d88eec96ca48e28225e4795b4e93e6a265114c22755a348b', 'review/Toys_and_Games': '2e6157468ea97963098fe6b14749eda8d16a4443e718dcf6cbd0726c781fd938', 'review/Unknown': 'c1f58c580a747250877369b603e8959985b11ea079eddc88f66d1144e08d917b', 'review/Video_Games': '7bde9e0a039bdb321e993bfe78a6c7d438fb1504b1ad152c0cb6a33a8b7cb26a'}


def _json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    tmp.replace(path)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False, allow_nan=False).encode('utf-8')).hexdigest()


def _quote(value: str | Path) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _official_url(kind: str, category: str) -> str:
    folder = 'review_categories' if kind == 'review' else 'meta_categories'
    name = category if kind == 'review' else 'meta_' + category
    return f'https://huggingface.co/datasets/{REPOSITORY}/resolve/{REVISION}/raw/{folder}/{name}.jsonl'


def manifest_from_verification(verification_path: str | Path, output_path: str | Path | None = None) -> dict:
    """Convert the archived publisher-tree verification into a pinned manifest."""
    old = json.loads(Path(verification_path).read_text(encoding='utf-8'))
    if old.get('commit') != REVISION:
        raise ValueError('Manifest revision differs from the audited pipeline revision')
    result = {'revision': REVISION, 'repository': REPOSITORY,
              'verification': 'publisher_file_metadata_not_full_row_scan',
              'review_entries': [], 'metadata_entries': []}
    for kind, folder, destination in [('review', 'review_categories', 'review_entries'),
                                      ('metadata', 'meta_categories', 'metadata_entries')]:
        for item in old['partitions'][folder]['files']:
            name = Path(item['path']).name
            category = name.removesuffix('.jsonl')
            if kind == 'metadata':
                category = category.removeprefix('meta_')
            key = kind + '/' + category
            if FIXED_SOURCE_SIZES.get(key) != item['size']:
                raise ValueError(f'Source file differs from frozen manifest: {key}')
            result[destination].append({'category': category, 'kind': kind, 'filename': name,
                'url': _official_url(kind, category), 'source_url': _official_url(kind, category),
                'expected_bytes': item['size'], 'expected_sha256': FIXED_SOURCE_SHA256[key], 'revision': REVISION})
        result[destination].sort(key=lambda x: x['category'])
    expected = sorted(k.split('/', 1)[1] for k in FIXED_SOURCE_SIZES if k.startswith('review/'))
    if [e['category'] for e in result['review_entries']] != expected:
        raise ValueError('Incomplete review manifest; all 34 category partitions are required')
    if len(result['metadata_entries']) != 34:
        raise ValueError('Incomplete metadata manifest')
    result['expected_review_categories'] = expected
    result['review_bytes'] = sum(e['expected_bytes'] for e in result['review_entries'])
    result['metadata_bytes'] = sum(e['expected_bytes'] for e in result['metadata_entries'])
    if output_path is not None:
        _json(Path(output_path), result)
    return result


def download_entry(entry: dict, raw_dir: str | Path, *, timeout: int = 90,
                   retries: int = 3) -> Path:
    """Download an immutable object, resuming only a matching partial file.

    A source that ignores a requested range is rejected before reading its body.
    Completed caches must match both the expected size and recorded SHA256.
    """
    raw_dir = Path(raw_dir); raw_dir.mkdir(parents=True, exist_ok=True)
    url = entry.get('url', entry.get('source_url'))
    expected = int(entry['expected_bytes'])
    if expected <= 0 or not isinstance(url, str) or not url.startswith('https://'):
        raise ValueError('Download requires an HTTPS URL and positive expected_bytes')
    filename = entry.get('filename', entry['category'] + '.jsonl')
    if Path(filename).name != filename:
        raise ValueError('Unsafe download filename')
    path = raw_dir / filename
    sidecar = path.with_suffix(path.suffix + '.download.json')
    signature = _hash({'url': url, 'bytes': expected, 'revision': entry.get('revision')})
    if path.exists():
        if not sidecar.exists():
            raise ValueError(f'Existing file lacks download audit: {path}')
        audit = json.loads(sidecar.read_text())
        if (audit.get('signature') != signature or path.stat().st_size != expected
                or sha256_file(path) != audit.get('sha256')
                or (entry.get('expected_sha256') and audit.get('sha256') != entry['expected_sha256'])):
            raise ValueError(f'Existing download failed verification: {path}')
        return path
    part = path.with_suffix(path.suffix + '.part')
    part_meta = path.with_suffix(path.suffix + '.part.json')
    if part.exists():
        if not part_meta.exists() or json.loads(part_meta.read_text()).get('signature') != signature:
            raise ValueError('Partial download belongs to another source; use a new directory')
        if part.stat().st_size > expected:
            raise ValueError('Partial download exceeds advertised source size')
    else:
        _json(part_meta, {'signature': signature, 'url': url, 'expected_bytes': expected})
    last_error = None
    for attempt in range(retries):
        offset = part.stat().st_size if part.exists() else 0
        if offset == expected:
            break
        headers = {'Accept-Encoding': 'identity'}
        if offset:
            headers['Range'] = f'bytes={offset}-'
        try:
            with requests.get(url, headers=headers, timeout=timeout, stream=True) as response:
                if offset:
                    if response.status_code != 206:
                        raise ValueError('Server did not honor HTTP Range; partial body was not read')
                    expected_range = f'bytes {offset}-{expected - 1}/{expected}'
                    if response.headers.get('Content-Range') != expected_range:
                        raise ValueError('Unexpected Content-Range; partial body was not read')
                elif response.status_code != 200:
                    response.raise_for_status()
                    raise ValueError(f'Expected complete HTTP 200, received {response.status_code}')
                content_length = response.headers.get('Content-Length')
                if content_length is not None and int(content_length) != expected - offset:
                    raise ValueError('Source byte length differs from frozen manifest')
                with part.open('ab' if offset else 'wb') as target:
                    for block in response.iter_content(chunk_size=1024 * 1024):
                        if not block: continue
                        if target.tell() + len(block) > expected:
                            raise ValueError('Source exceeded frozen manifest byte count')
                        target.write(block)
            if part.stat().st_size != expected:
                raise IOError('Truncated download; partial file retained for resume')
            break
        except (requests.RequestException, IOError, ValueError) as error:
            last_error = error
            if isinstance(error, ValueError):
                raise
            if attempt + 1 < retries:
                time.sleep(min(2 ** attempt, 4))
    if not part.exists() or part.stat().st_size != expected:
        raise IOError(f'Download incomplete after {retries} attempts: {last_error}')
    digest = sha256_file(part)
    if entry.get('expected_sha256') and digest != entry['expected_sha256']:
        raise ValueError('Downloaded SHA256 does not match expected source hash')
    part.replace(path)
    _json(sidecar, {'signature': signature, 'source_url': url, 'bytes': expected,
                   'sha256': digest, 'downloaded_at_utc': datetime.now(timezone.utc).isoformat()})
    part_meta.unlink(missing_ok=True)
    return path


@contextmanager
def _lock(directory: Path):
    # Target environments are macOS and Linux/SSH clusters.
    import fcntl
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / '.pipeline.lock').open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError('Output directory is already being processed') from error
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def _normalise(line: bytes, category: str, source_line: int):
    try:
        row = json.loads(line)
        if not isinstance(row, dict): return None, 'invalid_json'
        fingerprint = _hash(row)
    except (UnicodeError, ValueError, TypeError, json.JSONDecodeError):
        return None, 'invalid_json'
    ids = [row.get(k) for k in ['parent_asin', 'asin', 'user_id']]
    if any(not isinstance(x, str) or not x.strip() for x in ids):
        return None, 'missing_id'
    rating = row.get('rating')
    if isinstance(rating, bool) or not isinstance(rating, (int, float)) or not math.isfinite(rating) or rating not in [1, 2, 3, 4, 5]:
        return None, 'invalid_rating'
    timestamp = row.get('timestamp', row.get('sort_timestamp'))
    if (isinstance(timestamp, bool) or not isinstance(timestamp, int)
            or not MIN_TIMESTAMP <= timestamp < MAX_TIMESTAMP):
        return None, 'invalid_timestamp'
    day = datetime.fromtimestamp(timestamp // 1000, timezone.utc).date()
    week = day - timedelta(days=day.weekday())
    parent, asin, user = ids
    return {'record_hash': fingerprint, 'identity_hash': _hash([user, asin, timestamp]),
            'user_hash': hashlib.sha256(user.encode()).hexdigest(), 'parent_asin': parent,
            'asin': asin, 'timestamp_ms': timestamp, 'week': week, 'rating': int(rating),
            'verified_purchase': row.get('verified_purchase') if isinstance(row.get('verified_purchase'), bool) else None,
            'category': category, 'source_line': source_line}, 'accepted_for_dedup'


def _stage_source(source: dict, directory: Path, *, batch_size: int) -> dict:
    path = Path(source['path'])
    raw_sha = sha256_file(path)
    size = path.stat().st_size
    expected_size = source.get('expected_bytes')
    source_url = source.get('source_url', source.get('url'))
    pinned_source = source.get('revision') == REVISION and source_url == _official_url('review', source['category'])
    expected_hash = (FIXED_SOURCE_SHA256.get('review/' + source['category']) if pinned_source else source.get('expected_sha256'))
    if expected_hash and raw_sha != expected_hash:
        raise ValueError(f'Input source SHA256 does not match publisher LFS manifest: {path}')
    if expected_size is not None and size != int(expected_size):
        raise ValueError(f'Input source byte count does not match manifest: {path}')
    if source.get('expected_sha256') and raw_sha != source['expected_sha256']:
        raise ValueError(f'Input source SHA256 mismatch: {path}')
    source_signature = _hash({'pipeline': PIPELINE_VERSION, 'sha256': raw_sha,
                             'category': source['category'], 'batch_size': batch_size,
                             'source_url': source.get('source_url', source.get('url')),
                             'revision': source.get('revision'), 'expected_bytes': expected_size,
                             'expected_sha256': expected_hash})
    directory.mkdir(parents=True, exist_ok=True)
    audit_path = directory / 'stage_audit.json'
    if audit_path.exists():
        saved = json.loads(audit_path.read_text())
        if saved['signature'] != source_signature:
            raise ValueError('Staging cache has different input or parameters; use a new output directory')
        if all((directory / name).is_file() and sha256_file(directory / name) == digest
               for name, digest in saved['parquet_sha256'].items()):
            return saved
        raise ValueError('Staging cache SHA256 mismatch')
    # An interrupted source is restarted; completed other sources remain intact.
    for old in directory.glob('part-*.parquet'):
        old.unlink()
    counts = Counter({key: 0 for key in FILTER_ORDER})
    batch = []; part = 0; physical_lines = 0; blank_lines = 0; metadata = {}
    def flush():
        nonlocal part, batch
        if not batch: return
        target = directory / f'part-{part:06d}.parquet'
        pq.write_table(pa.Table.from_pylist(batch, schema=STAGE_SCHEMA), target, compression='zstd')
        metadata[target.name] = sha256_file(target)
        part += 1; batch = []
    with path.open('rb') as stream:
        for number, line in enumerate(stream, 1):
            physical_lines += 1
            if not line.strip():
                blank_lines += 1; continue
            clean, reason = _normalise(line, source['category'], number)
            counts[reason] += 1
            if clean is not None: batch.append(clean)
            if len(batch) >= batch_size: flush()
            if physical_lines % 1000000 == 0:
                print(json.dumps({'source': source['category'], 'physical_lines_scanned': physical_lines}), flush=True)
    flush()
    if not metadata:
        target = directory / 'part-000000.parquet'
        pq.write_table(pa.Table.from_pylist([], schema=STAGE_SCHEMA), target)
        metadata[target.name] = sha256_file(target)
    url = source.get('source_url', source.get('url'))
    category = source['category']
    official = (source.get('revision') == REVISION and url == _official_url('review', category)
                and size == FIXED_SOURCE_SIZES.get('review/' + category)
                and expected_size == size and raw_sha == FIXED_SOURCE_SHA256.get('review/' + category)
                and source.get('kind', 'review') == 'review')
    audit = {'signature': source_signature, 'source_category': category, 'source_name': path.name,
             'source_url': url, 'revision': source.get('revision'), 'source_bytes': size,
             'source_sha256': raw_sha, 'publisher_expected_sha256': expected_hash, 'source_identity_verified': official,
             'physical_lines': physical_lines, 'blank_lines': blank_lines,
             'scanned_rows': physical_lines - blank_lines, 'filter_ledger': dict(counts),
             'filter_order': FILTER_ORDER, 'parquet_sha256': metadata,
             'created_at_utc': datetime.now(timezone.utc).isoformat()}
    assert sum(counts.values()) == audit['scanned_rows']
    _json(audit_path, audit)
    return audit


def _connection(directory: Path, memory_limit: str, threads: int):
    database = directory / 'working.duckdb'
    con = duckdb.connect(str(database))
    con.execute(f'SET memory_limit={_quote(memory_limit)}')
    con.execute(f'SET threads={int(threads)}')
    con.execute(f'SET temp_directory={_quote(directory / "spill")}')
    con.execute('SET preserve_insertion_order=false')
    return con


def _sample_condition(fraction: float, seed: int) -> str:
    if not 0 < fraction <= 1 or not math.isfinite(fraction):
        raise ValueError('sample_fraction must lie in (0,1]')
    boundary = math.floor(fraction * (1 << 60))
    return f"CAST('0x' || substr(md5(user_hash || {_quote('|' + str(seed))}),1,15) AS BIGINT) < {boundary}"


def _bucket_expression(buckets: int):
    return f"lpad(CAST(CAST('0x' || substr(md5(parent_asin),1,8) AS BIGINT) % {int(buckets)} AS VARCHAR),3,'0')"


def _export(con, table: str, output_dir: Path, buckets: int) -> dict:
    """Disk-backed export; category is an attribute, never a panel count key."""
    build = output_dir / '_finalizing'
    if build.exists(): shutil.rmtree(build)
    build.mkdir()
    (build / 'clean').mkdir(); (build / 'panel').mkdir()
    bucket = _bucket_expression(buckets)
    if con.execute(f'SELECT count(*) FROM {table}').fetchone()[0]:
        con.execute(f'COPY (SELECT *, {bucket} AS bucket FROM {table}) TO {_quote(build / "clean")} '
                    '(FORMAT PARQUET, PARTITION_BY(bucket), COMPRESSION ZSTD)')
        query = f'''SELECT parent_asin,asin,week,
                    count(*) FILTER (WHERE rating=1)::BIGINT AS n1,
                    count(*) FILTER (WHERE rating=2)::BIGINT AS n2,
                    count(*) FILTER (WHERE rating=3)::BIGINT AS n3,
                    count(*) FILTER (WHERE rating=4)::BIGINT AS n4,
                    count(*) FILTER (WHERE rating=5)::BIGINT AS n5,
                    count(*)::BIGINT AS n,
                    count(*) FILTER (WHERE verified_purchase=true)::BIGINT AS verified_n,
                    min(category) AS category, {bucket} AS bucket
                    FROM {table} GROUP BY parent_asin,asin,week'''
        con.execute(f'COPY ({query}) TO {_quote(build / "panel")} '
                    '(FORMAT PARQUET, PARTITION_BY(bucket), COMPRESSION ZSTD)')
    for name in ['clean', 'panel']:
        target = output_dir / name
        if target.exists(): shutil.rmtree(target)
        (build / name).replace(target)
    build.rmdir()
    files = {str(p.relative_to(output_dir)): sha256_file(p)
             for folder in ['clean', 'panel'] for p in sorted((output_dir / folder).rglob('*.parquet'))}
    panels = sorted((output_dir / 'panel').rglob('*.parquet'))
    panel_rows = sum(pq.ParquetFile(p).metadata.num_rows for p in panels)
    return {'artifact_sha256': files, 'panel_rows': panel_rows,
            'panel_buckets': [int(p.name.split('=')[1]) for p in sorted((output_dir / 'panel').glob('bucket=*'))]}


def _validate_audit_header(audit: dict) -> None:
    ledger = audit['filter_ledger']
    if any(not isinstance(value, int) or value < 0 for value in ledger.values()):
        raise ValueError('Audit ledger contains invalid counts')
    if audit['scanned_rows'] != sum(ledger.values()) or audit['accepted_rows'] != ledger['accepted']:
        raise ValueError('Source-row filter ledger does not conserve counts')
    if audit['panel_rows'] > audit['accepted_rows']:
        raise ValueError('Panel rows exceed accepted review count')
    artifact_buckets = {'clean': set(), 'panel': set()}
    for name in audit['artifact_sha256']:
        parts = Path(name).parts
        if (len(parts) != 3 or parts[0] not in artifact_buckets
                or not parts[1].startswith('bucket=') or not parts[2].endswith('.parquet')):
            raise ValueError('Unsafe or unexpected artifact path in audit')
        label = parts[1].removeprefix('bucket=')
        if len(label) != 3 or not label.isdigit() or not 0 <= int(label) < audit['params']['buckets']:
            raise ValueError('Artifact bucket is outside configured partition range')
        artifact_buckets[parts[0]].add(int(label))
    recorded = audit['panel_buckets']
    if (any(type(b) is not int for b in recorded)
            or recorded != sorted(set(recorded))
            or recorded != sorted(artifact_buckets['panel'])
            or artifact_buckets['panel'] != artifact_buckets['clean']):
        raise ValueError('panel_buckets do not match the clean/panel artifact manifest')
    if bool(audit['accepted_rows']) != bool(recorded):
        raise ValueError('Accepted review count is inconsistent with exported buckets')
    if audit.get('coverage') == 'clean_review_rows_resampled_not_raw_jsonl_rescan':
        if audit.get('full_published_reviews_processed'):
            raise ValueError('Resampled clean rows cannot establish complete raw corpus processing')
        return
    sources = audit['sources']; params = audit['params']
    if len(sources) != audit['source_count'] or len({x['source_category'] for x in sources}) != len(sources):
        raise ValueError('Audit source categories are duplicated or inconsistent')
    if sum(x['scanned_rows'] for x in sources) != audit['scanned_rows']:
        raise ValueError('Source audits do not reproduce scanned rows')
    for source in sources:
        if sum(source['filter_ledger'].values()) != source['scanned_rows']:
            raise ValueError('Source filtering ledger is inconsistent')
    valid = sum(x['filter_ledger']['accepted_for_dedup'] for x in sources)
    if (valid != audit['valid_rows_before_deduplication']
            or valid - ledger['exact_duplicate'] != audit['exact_unique_rows']
            or audit['exact_unique_rows'] - ledger['identity_conflict'] != audit['clean_rows_before_sampling']
            or audit['clean_rows_before_sampling'] - ledger['sampling_excluded'] != audit['accepted_rows']):
        raise ValueError('Audit count layers are inconsistent')
    signature = _hash({'params': params, 'sources': [x['signature'] for x in sources],
                       'source_identity': [{k:x[k] for k in ['source_url','revision','source_identity_verified']}
                                           for x in sources]})
    if signature != audit['signature']:
        raise ValueError('Audit parameters or source signature changed')
    expected_categories = {k.split('/', 1)[1] for k in FIXED_SOURCE_SIZES if k.startswith('review/')}
    proven = ({x['source_category'] for x in sources} == expected_categories
              and params['sample_fraction'] == 1.0)
    for source in sources:
        category = source['source_category']
        proven = proven and (source.get('source_identity_verified')
            and source['revision'] == REVISION and source['source_url'] == _official_url('review', category)
            and source['source_bytes'] == FIXED_SOURCE_SIZES.get('review/' + category)
            and source['source_sha256'] == FIXED_SOURCE_SHA256.get('review/' + category))
    if bool(audit['full_published_reviews_processed']) != bool(proven):
        raise ValueError('Full-corpus claim is not supported by the frozen source manifest')


def _validate_audit(directory: Path, audit: dict, *, prefix: str | None = None) -> None:
    _validate_audit_header(audit)
    for name, expected in audit['artifact_sha256'].items():
        if Path(name).is_absolute() or '..' in Path(name).parts or not name.startswith(('clean/', 'panel/')):
            raise ValueError('Unsafe artifact path in audit')
        if prefix is not None and not name.startswith(prefix): continue
        path = directory / name
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f'Output artifact SHA256 mismatch: {name}')
    ledger = audit['filter_ledger']
    if audit['scanned_rows'] != sum(ledger.values()):
        raise ValueError('Source-row filter ledger does not conserve counts')


def process_sources(sources: Iterable[dict], output_dir: str | Path, *, buckets: int = 32,
                    batch_size: int = 100000, memory_limit: str = '4GB', threads: int = 4,
                    sample_fraction: float = 1.0, sample_seed: int = 42) -> dict:
    """Scan each source completely, deduplicate globally, quarantine conflicts.

    Staging checkpoints are per source. A completed output is reusable only if
    all input hashes, parameters, and exported artifact hashes still agree.
    """
    sources = [dict(s) for s in sources]
    if not sources or not 1 <= buckets <= 999 or batch_size < 1 or threads < 1:
        raise ValueError('Need nonempty sources and positive batch_size/threads, 1..999 buckets')
    if len({s['category'] for s in sources}) != len(sources):
        raise ValueError('Duplicate source category; each selected source must occur once')
    condition = _sample_condition(sample_fraction, sample_seed)
    output = Path(output_dir)
    params = {'pipeline_version': PIPELINE_VERSION, 'buckets': buckets, 'batch_size': batch_size,
              'sample_fraction': sample_fraction, 'sample_seed': sample_seed,
              'timestamp_min_inclusive': MIN_TIMESTAMP, 'timestamp_max_exclusive': MAX_TIMESTAMP,
              'duplicate_policy': 'canonical_full_json_global_sha256',
              'conflict_policy': 'quarantine_all_distinct_records_per_user_asin_timestamp',
              'week_policy': 'UTC_Monday', 'memory_limit': memory_limit, 'threads': threads}
    started = time.monotonic()
    with _lock(output):
        stages = []
        for source in sorted(sources, key=lambda s: s['category']):
            key = hashlib.sha256(source['category'].encode()).hexdigest()[:20]
            stages.append(_stage_source(source, output / 'staging' / key, batch_size=batch_size))
        signature = _hash({'params': params, 'sources': [s['signature'] for s in stages],
                           'source_identity': [{k:s[k] for k in ['source_url','revision','source_identity_verified']}
                                               for s in stages]})
        audit_file = output / 'audit.json'
        if audit_file.exists():
            previous = json.loads(audit_file.read_text())
            if previous.get('signature') != signature:
                raise ValueError('Completed output has different sources or parameters; use a new output directory')
            _validate_audit(output, previous)
            return previous
        staging_paths = [str(p) for p in sorted((output / 'staging').glob('*/part-*.parquet'))]
        # Only the explicitly selected current stages, never unrelated files.
        selected_dirs = {hashlib.sha256(s['category'].encode()).hexdigest()[:20] for s in sources}
        staging_paths = [p for p in staging_paths if Path(p).parent.name in selected_dirs]
        read = 'read_parquet([' + ','.join(_quote(p) for p in staging_paths) + '],hive_partitioning=false)'
        con = _connection(output, memory_limit, threads)
        try:
            con.execute(f'''CREATE OR REPLACE TABLE deduplicated AS
                SELECT * EXCLUDE(rn) FROM (SELECT *, row_number() OVER
                    (PARTITION BY record_hash ORDER BY category,source_line) rn FROM {read}) WHERE rn=1''')
            valid_before = sum(s['filter_ledger']['accepted_for_dedup'] for s in stages)
            unique = con.execute('SELECT count(*) FROM deduplicated').fetchone()[0]
            con.execute('''CREATE OR REPLACE TABLE conflict_keys AS SELECT identity_hash FROM deduplicated
                           GROUP BY identity_hash HAVING count(*)>1''')
            conflict_groups = con.execute('SELECT count(*) FROM conflict_keys').fetchone()[0]
            conflict_rows = con.execute('SELECT count(*) FROM deduplicated SEMI JOIN conflict_keys USING(identity_hash)').fetchone()[0]
            con.execute('''CREATE OR REPLACE TABLE clean_all AS SELECT * FROM deduplicated
                           ANTI JOIN conflict_keys USING(identity_hash)''')
            clean_before = con.execute('SELECT count(*) FROM clean_all').fetchone()[0]
            multi_parent = con.execute('''SELECT count(*) FROM (SELECT asin FROM clean_all
                                        GROUP BY asin HAVING count(DISTINCT parent_asin)>1)''').fetchone()[0]
            con.execute(f'CREATE OR REPLACE TABLE clean_selected AS SELECT * FROM clean_all WHERE {condition}')
            accepted = con.execute('SELECT count(*) FROM clean_selected').fetchone()[0]
            export = _export(con, 'clean_selected', output, buckets)
            con.execute('DROP TABLE clean_selected'); con.execute('DROP TABLE clean_all')
            con.execute('DROP TABLE conflict_keys'); con.execute('DROP TABLE deduplicated')
            con.execute('CHECKPOINT')
        finally:
            con.close()
        (output / 'working.duckdb').unlink(missing_ok=True)
        ledger = {k: sum(s['filter_ledger'][k] for s in stages) for k in FILTER_ORDER[:-1]}
        ledger.update({'exact_duplicate': valid_before - unique, 'identity_conflict': conflict_rows,
                       'sampling_excluded': clean_before - accepted, 'accepted': accepted})
        scanned = sum(s['scanned_rows'] for s in stages)
        assert scanned == sum(ledger.values())
        expected_categories = {k.split('/', 1)[1] for k in FIXED_SOURCE_SIZES if k.startswith('review/')}
        full = ({s['source_category'] for s in stages} == expected_categories
                and all(s['source_identity_verified'] for s in stages) and sample_fraction == 1.0)
        audit = {'pipeline_version': PIPELINE_VERSION, 'signature': signature, 'params': params,
                 'created_at_utc': datetime.now(timezone.utc).isoformat(),
                 'source_count': len(stages), 'sources': stages, 'scanned_rows': scanned,
                 'physical_lines': sum(s['physical_lines'] for s in stages),
                 'blank_lines': sum(s['blank_lines'] for s in stages),
                 'valid_rows_before_deduplication': valid_before, 'exact_unique_rows': unique,
                 'identity_conflict_groups': conflict_groups, 'clean_rows_before_sampling': clean_before,
                 'accepted_rows': accepted, 'filter_ledger': ledger,
                 'mapping_warning_child_ids_with_multiple_parents': multi_parent,
                 'mapping_policy': 'Keep supplied parent-child pairs; do not infer historical relationships.',
                 'full_published_reviews_processed': full, 'metadata_display_processed': False,
                 'coverage': 'full_34_published_review_files' if full else 'selected_sources_fully_scanned',
                 'sampling_kind': 'none' if sample_fraction == 1 else 'deterministic_nested_reviewer_sampling',
                 'scope': 'Published observable reviews, not all Amazon reviews or all buyers.',
                 'elapsed_seconds': round(time.monotonic() - started, 3), **export}
        _json(audit_file, audit)
        return audit


def list_panel_buckets(output_dir: str | Path) -> list[int]:
    audit = json.loads((Path(output_dir) / 'audit.json').read_text())
    _validate_audit_header(audit)
    return audit['panel_buckets']


def load_panel_bucket(output_dir: str | Path, bucket: int) -> pd.DataFrame:
    directory = Path(output_dir)
    audit = json.loads((directory / 'audit.json').read_text())
    prefix = f'panel/bucket={int(bucket):03d}/'
    names = sorted(n for n in audit['artifact_sha256'] if n.startswith(prefix))
    _validate_audit(directory, audit, prefix=prefix)
    if not names:
        return pd.DataFrame(columns=PANEL_COLUMNS)
    frames = [pq.ParquetFile(directory / name).read().to_pandas() for name in names]
    result = pd.concat(frames, ignore_index=True)
    if result.duplicated(['parent_asin', 'asin', 'week']).any():
        raise ValueError('Duplicate parent/child/week panel keys')
    if (result[[f'n{i}' for i in range(1, 6)]].sum(axis=1) != result.n).any():
        raise ValueError('Panel rating counts do not add to n')
    return result


def sample_clean(output_dir: str | Path, target_dir: str | Path, fraction: float,
                 seed: int = 42, *, memory_limit: str = '4GB', threads: int = 4) -> dict:
    """Re-read CLEAN review rows and reaggregate nested reviewer samples.

    This is not a second scan of original JSONL and not an end-to-end ingest
    benchmark. Sampling from a previously sampled parent output is prohibited.
    """
    origin = Path(output_dir); target = Path(target_dir)
    if origin.resolve() == target.resolve():
        raise ValueError('Sample output must differ from input')
    parent = json.loads((origin / 'audit.json').read_text())
    if parent.get('params', {}).get('sample_fraction') != 1.0:
        raise ValueError('Reviewer scale experiments require unsampled clean source rows')
    _validate_audit(origin, parent, prefix='clean/')
    condition = _sample_condition(fraction, seed)
    paths = [origin / n for n in parent['artifact_sha256'] if n.startswith('clean/')]
    if not paths:
        raise ValueError('No clean rows are available for sampling')
    signature = _hash({'source_signature': parent['signature'], 'fraction': fraction, 'seed': seed})
    with _lock(target):
        if (target / 'audit.json').exists():
            old = json.loads((target / 'audit.json').read_text())
            if old.get('signature') != signature:
                raise ValueError('Sampling output parameters changed; use a new directory')
            _validate_audit(target, old); return old
        con = _connection(target, memory_limit, threads)
        try:
            read = 'read_parquet([' + ','.join(_quote(p) for p in paths) + '],hive_partitioning=false)'
            con.execute(f'CREATE OR REPLACE TABLE sampled AS SELECT * FROM {read} WHERE {condition}')
            accepted = con.execute('SELECT count(*) FROM sampled').fetchone()[0]
            exported = _export(con, 'sampled', target, parent['params']['buckets'])
            con.execute('DROP TABLE sampled'); con.execute('CHECKPOINT')
        finally:
            con.close()
        (target / 'working.duckdb').unlink(missing_ok=True)
        scanned = parent['accepted_rows']
        audit = {'pipeline_version': PIPELINE_VERSION, 'signature': signature,
                 'params': {'sample_fraction': fraction, 'sample_seed': seed, 'buckets': parent['params']['buckets']},
                 'scanned_rows': scanned, 'accepted_rows': accepted,
                 'filter_ledger': {'sampling_excluded': scanned - accepted, 'accepted': accepted},
                 'full_published_reviews_processed': False,
                 'coverage': 'clean_review_rows_resampled_not_raw_jsonl_rescan',
                 'sampling_kind': 'deterministic_nested_reviewer_sampling',
                 'source_audit_signature': parent['signature'],
                 'interpretation': 'Cluster-preserving row reaggregation; not raw ingest speed or the full Amazon population.',
                 'created_at_utc': datetime.now(timezone.utc).isoformat(), **exported}
        _json(target / 'audit.json', audit)
        return audit
