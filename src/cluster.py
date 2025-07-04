import dask
from dask.distributed import Client, LocalCluster

import logging
import logging.config

logger = logging.getLogger(__name__)

def start_dask_cluster():

    cluster = LocalCluster()
    client = Client(cluster)

    logger.info("Initialized Dask client on: %s", client)
    logger.info(f"Dashboard available at: {client.dashboard_link}")
    logger.info(f"Chunksize specified at {dask.config.get('array.chunk-size')}")

    return(client, cluster)