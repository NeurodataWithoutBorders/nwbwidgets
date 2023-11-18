from dandi.dandiapi import DandiAPIClient


def get_all_dandisets_metadata():
    with DandiAPIClient() as client:
        all_metadata = list()
        for ii, m in enumerate(client.get_dandisets()):
            if ii > 150 and ii < 160:
                try:
                    all_metadata.append(m.get_metadata())
                except:
                    pass
            else:
                pass
    return all_metadata


def get_dandiset_metadata(dandiset_id: str):
    with DandiAPIClient() as client:
        dandiset = client.get_dandiset(dandiset_id=dandiset_id, version_id="draft")
        return dandiset.get_metadata()


def list_dandiset_files(dandiset_id: str):
    with DandiAPIClient() as client:
        dandiset = client.get_dandiset(dandiset_id=dandiset_id, version_id="draft")
        return [i.dict().get("path") for i in dandiset.get_assets()]


def get_file_url(dandiset_id: str, file_path: str):
    with DandiAPIClient() as client:
        asset = client.get_dandiset(dandiset_id, "draft").get_asset_by_path(file_path)
        return asset.get_content_url(follow_redirects=1, strip_query=True)
