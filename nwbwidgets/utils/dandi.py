from dandi.dandiapi import DandiAPIClient
from dandischema.models import Dandiset


def get_dandiset_metadata(dandiset_id: str):
    with DandiAPIClient() as client:
        dandiset = client.get_dandiset(dandiset_id=dandiset_id, version_id="draft")
        return dandiset.get_metadata()


def list_dandiset_files(dandiset_id: str):
    with DandiAPIClient() as client:
        dandiset = client.get_dandiset(dandiset_id=dandiset_id, version_id="draft")
        return sorted([i.dict().get("path") for i in dandiset.get_assets() if i.dict().get("path").endswith(".nwb")])


def get_file_url(dandiset_id: str, file_path: str):
    with DandiAPIClient() as client:
        asset = client.get_dandiset(dandiset_id, "draft").get_asset_by_path(file_path)
        return asset.get_content_url(follow_redirects=1, strip_query=True)


def has_nwb(metadata: Dandiset):
    if hasattr(metadata, "assetsSummary"):
        assets_summary = metadata.assetsSummary
        if hasattr(assets_summary, "dataStandard"):
            return any(x.identifier == "RRID:SCR_015242" for x in assets_summary.dataStandard)
    return False
