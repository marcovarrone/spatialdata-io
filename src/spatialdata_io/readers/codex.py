from __future__ import annotations

import os
import re
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any

import anndata as ad
import pandas as pd
import readfcs
from dask_image.imread import imread
from spatialdata import SpatialData, read_zarr
from spatialdata._logging import logger
from spatialdata.models import Image2DModel, ShapesModel, TableModel

from spatialdata_io._constants._constants import CodexKeys
from spatialdata_io._docs import inject_docs

__all__ = ["codex"]


@inject_docs(vx=CodexKeys)
def codex(
    path: str | Path,
    fcs: bool = True,
    imread_kwargs: Mapping[str, Any] = MappingProxyType({}),
    # output_path: Path | None = None, # ADDED_BY_F
) -> SpatialData:
    """
    Read *CODEX* formatted dataset.

    This function reads the following files:

        - ``<dataset_id>_`{vx.FCS_FILE!r}```: Counts and metadata file.
        - ``<dataset_id>_`{vx.IMAGE_TIF!r}```: High resolution tif image.

    .. seealso::

        - `CODEX output <https://help.codex.bio/codex/processor/technical-notes/expected-output>`_.

    Parameters
    ----------
    path
        Path to the directory containing the data.
    fcs
        Whether a .fcs file is provided. If False, a .csv file is expected.
    imread_kwargs
        Keyword arguments passed to :func:`dask_image.imread.imread`.
    output_path
        Path to directly write every element to a zarr file as soon as it is read. This can decrease the memory requirement.

    Returns
    -------
    :class:`spatialdata.SpatialData`
    """
    path = Path(path)
    # output_path = Path(output_path) if output_path is not None else None # ADDED_BY_F
    patt = re.compile(".*.fcs") if fcs else re.compile(".*.csv")
    path_files = [i for i in os.listdir(path) if patt.match(i)]
    if path_files and CodexKeys.FCS_FILE or CodexKeys.FCS_FILE_CSV in patt.pattern:
        fcs = (
            readfcs.ReadFCS(path / path_files[0]).data
            if CodexKeys.FCS_FILE in path_files[0]
            else pd.read_csv(path_files[0], header=0, index_col=None)
        )
    else:
        raise ValueError("Cannot determine data set. Expecting a file with format .fcs or .csv")

    adata = _codex_df_to_anndata(fcs)

    # sdata = SpatialData()
    # if output_path is not None:
    #     sdata.write(output_path)

    xy = adata.obsm[CodexKeys.SPATIAL_KEY]
    shapes = ShapesModel.parse(xy, geometry=0, radius=1, index=adata.obs[CodexKeys.INSTANCE_KEY])
    region = adata.obs[CodexKeys.REGION_KEY].unique()[0]
    adata.obs[CodexKeys.REGION_KEY] = adata.obs[CodexKeys.REGION_KEY].astype("category")
    table = TableModel.parse(adata, region=region, region_key=CodexKeys.REGION_KEY, instance_key=CodexKeys.INSTANCE_KEY)
    

    im_patt = re.compile(".*.tif")
    path_files = [i for i in os.listdir(path) if im_patt.match(i)]
    if path_files and CodexKeys.IMAGE_TIF in path_files[0]:
        image = imread(path_files[0], **imread_kwargs)
        images = {
            "images": Image2DModel.parse(
                image,
                scale_factors=[2, 2],
                rgb=None,
            )
        }
        sdata = SpatialData(images=images, shapes={str(region): shapes}, table=table)
    else:
        logger.warning("Cannot find .tif file. Will build spatialdata with shapes and table only.")
        sdata = SpatialData(shapes={str(region): shapes}, table=table)

    return sdata


def _codex_df_to_anndata(df: pd.DataFrame) -> ad.AnnData:
    """Convert a codex formatted .fcs dataframe or .csv file to anndata."""
    adata = ad.AnnData(df.filter(regex="cyc.*"))
    adata.obs = df[df.columns.drop(list(df.filter(regex="cyc.*")))]
    adata.obsm[CodexKeys.SPATIAL_KEY] = df[["x", "y"]].values
    adata.var_names_make_unique()
    return adata
