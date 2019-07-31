import argparse
import logging
import multiprocessing
import os.path
import random
import shutil
import string
import subprocess
import sys
import tempfile
from typing import Union

import gdal2tiles
import geojson
import mbutil
import shapely.wkt
from gdalconst import GA_ReadOnly
from osgeo import gdal, osr


def build_cmd_line_args():
    parser = argparse.ArgumentParser(description='Merge a bunch of GeoPDFs and convert them')
    parser.add_argument('--verbose', help='Output extra logging at DEBUG level', action='store_true')
    parser.add_argument('--quiet', help='Limit the amount of debug information', action='store_true')
    parser.add_argument('--geopdf', help='GeoPDF file to convert', type=str, required=True)
    parser.add_argument('--output-directory', type=str, help='Where to store the output file', required=True)
    parser.add_argument('--overwrite', help='If the destination file exists then overwrite it', action='store_true')
    parser.add_argument('--name', type=str, help='Name of converted file, if blank will use the geopdf name.')
    parser.add_argument('--output-format', choices=['GTiff', 'MBTiles', 'XYZ', 'TMS'], default='GTiff', type=str)
    parser.add_argument('--workspace', help='Where to put working files, will use temp diretory if blank', type=str)
    parser.add_argument('--temp-directory', help='Set the temp directory location', type=str)
    parser.add_argument('--resize-tiff', help='Resizse the Tiff to be evenly divisible by tile size',
                        action='store_true')

    return parser.parse_args()


def pdf_layers_to_include():
    return ['Map_Frame.Geographic_Names',
            'Map_Frame.Structures',
            'Map_Frame.Transportation.Road_Names_and_Shields',
            'Map_Frame.Transportation.Road_Features',
            'Map_Frame.Transportation.Trails',
            'Map_Frame.Transportation.Railroads',
            'Map_Frame.Transportation.Airports',
            'Map_Frame.PLSS',
            'Map_Frame.Wetlands',
            'Map_Frame.Hydrography',
            'Map_Frame.Terrain.Contours',
            'Map_Frame.Woodland',
            'Map_Frame.Boundaries.Jurisdictional_Boundaries.International',
            'Map_Frame.Boundaries.Jurisdictional_Boundaries.State_or_Territory',
            'Map_Frame.Boundaries.Jurisdictional_Boundaries.County_or_Equivalent',
            'Map_Frame.Boundaries.Federal_Administered_Lands.National_Cemetery',
            'Map_Frame.Boundaries.Federal_Administered_Lands.National_Park_Service',
            'Map_Frame.Boundaries.Federal_Administered_Lands.Department_of_Defense',
            'Map_Frame.Boundaries.Federal_Administered_Lands.Forest_Service']


def get_raster_size(gdal_ds: gdal.Dataset) -> tuple:
    return gdal_ds.RasterXSize, gdal_ds.RasterYSize


def find_neatline(geotiff_path: str) -> Union[geojson.Feature, None]:
    log_message_with_border('Finding the neatline from the GeoTIFF')

    if not os.path.exists(geotiff_path):
        logger.error('{} does not exist'.format(geotiff_path))
        sys.exit(1)

    srs_ds = gdal.Open(geotiff_path, GA_ReadOnly)

    try:
        if srs_ds:
            info = gdal.Info(srs_ds, format='json')

            if info:
                neatline_wkt = info['metadata']['']['NEATLINE']

                wkt = shapely.wkt.loads(neatline_wkt)
                neatline_geojson = geojson.Feature(geometry=wkt, properties={})

                return neatline_geojson
    finally:
        close_raster_dataset(srs_ds)

    return None


def add_srs_to_geojson(neatline_geojson: geojson.Feature, srs: str, workspace: str) -> str:
    log_message_with_border('Adding srs information to GeoJSON neatline')

    if not neatline_geojson:
        logger.error('neatline_geojson is empty')
        sys.exit(1)

    if not srs:
        logger.error('srs is empty')
        sys.exit(1)

    unprojected_geojson_path = create_tempfile_path(workspace, 'unprojected_geojson', 'json')
    neatline_geojson_path = create_tempfile_path(workspace, 'neatline_geojson', 'json')

    with open(unprojected_geojson_path, 'w') as fp:
        geojson.dump(neatline_geojson, fp)

    subprocess.check_output(['ogr2ogr', '-f', 'GeoJSON', '-a_srs', srs,
                             neatline_geojson_path, unprojected_geojson_path])

    with open(neatline_geojson_path, 'r') as fp:
        neatline_geojson = geojson.load(fp)
        logger.debug('Result from trying to add srs to neatline={}'.format(geojson.dumps(neatline_geojson)))

    return neatline_geojson_path


def find_srs(geotiff_path: str) -> Union[str, None]:
    log_message_with_border('Finding the SRS of GeoTIFF')

    if not os.path.exists(geotiff_path):
        logger.error('{} does not exist'.format(geotiff_path))
        sys.exit(1)

    srs_ds = gdal.Open(geotiff_path, GA_ReadOnly)

    try:
        if srs_ds:
            info = gdal.Info(srs_ds, format='json')

            if info:
                logger.info(info)
                coordinate_system_wkt = info['coordinateSystem']['wkt']
                if coordinate_system_wkt:
                    srs = osr.SpatialReference()
                    srs.ImportFromWkt(coordinate_system_wkt)

                    return '{}:{}'.format(srs.GetAuthorityName(None), srs.GetAuthorityCode(None))
    finally:
        close_raster_dataset(srs_ds)

    return None


def create_tempfile_path(workspace: str, filename: str, extension: str) -> str:
    temp = os.path.join(workspace,
                        '{}_{}.{}'.format(filename, ''.join([random.choice(string.ascii_letters) for n in range(25)]),
                                          extension))

    return temp


def log_message_with_border(message):
    border_length = len(message) + 4

    logger.info(
        """
        {}
        {}
        {}""".format('-' * border_length, '| {} |'.format(message), '-' * border_length))


def convert_to_geotiff(geopdf_path: str, gdal_pdf_layers: str, workspace: str, block_size=256,
                       gdal_pdf_dpi=600) -> [gdal.Dataset, str]:
    log_message_with_border('Converting GeoPDF to GeoTIFF')

    gdal.SetConfigOption('GDAL_PDF_DPI', str(gdal_pdf_dpi))
    gdal.SetConfigOption('GDAL_PDF_LAYERS', gdal_pdf_layers)

    create_options = ['NUM_THREADS=ALL_CPUS', 'BLOCKXSIZE={}'.format(block_size), 'BLOCKYSIZE={}'.format(block_size)]

    options = gdal.TranslateOptions(
        format='GTiff',
        creationOptions=create_options,
        resampleAlg='lanczos')

    geotiff_path = create_tempfile_path(workspace, 'unclipped_geotiff', 'tif')

    geotiff_ds = gdal.Translate(geotiff_path, geopdf_path, options=options)

    logger.debug('Result from converting GeoPDF to GeoTIFF={}'.format(geotiff_ds))

    return geotiff_ds, geotiff_path


def crop_geotiff(geotiff_ds: gdal.Dataset, neatline_geojson_path: str, workspace: str) -> [gdal.Dataset, str]:
    log_message_with_border('Crop GeoTIFF using neatline polygon')

    if not os.path.exists(neatline_geojson_path):
        logger.error('{} does not exist'.format(neatline_geojson_path))
        sys.exit(1)

    clipped_geotiff_path = create_tempfile_path(workspace, 'clipped_geotiff', 'tif')

    create_options = ['TILED=YES']

    options = gdal.WarpOptions(cropToCutline=True,
                               creationOptions=create_options,
                               multithread=True,
                               warpMemoryLimit=8000,
                               warpOptions=['NUM_THREADS=ALL_CPUS'],
                               cutlineDSName=neatline_geojson_path,
                               dstSRS='EPSG:3857',
                               srcAlpha=True)

    return gdal.Warp(clipped_geotiff_path, geotiff_ds, options=options), clipped_geotiff_path


def close_raster_dataset(ds: gdal.Dataset) -> None:
    if ds:
        ds = None


def cleanup(workspace: str) -> None:
    log_message_with_border('Cleaning up workspace')

    logger.debug('Deleting working files from {}'.format(workspace))

    for file in os.listdir(workspace):
        logger.info('Deleting {} from {} [{}]'.format(file, workspace, file_size(os.path.join(workspace, file))))

    logger.info('Deleting workspace {}'.format(workspace))

    shutil.rmtree(workspace)


def convert_bytes(num):
    """
    this function will convert bytes to MiB.... GiB... etc
    """
    for x in ['bytes', 'KiB', 'MiB', 'GiB', 'TiB']:
        if num < 1024.0:
            return f'{num:.1f} {x}'
        num /= 1024.0


def file_size(file_path):
    """
    this function will return the file size
    """
    if os.path.isfile(file_path):
        file_info = os.stat(file_path)
        return convert_bytes(file_info.st_size)


def reproject_geotiff(geotiff_ds: gdal.Dataset, workspace: str, source_srs: str, target_srs: str) -> [gdal.Dataset,
                                                                                                      str]:
    log_message_with_border('Changing projection of GeoTIFF')

    logger.debug('Changing srs from {} to {}'.format(source_srs, target_srs))

    reprojected_geotiff_path = create_tempfile_path(workspace, 'reprojected_geotiff', 'tif')

    create_options = ['TILED=YES', 'BIGTIFF=IF_NEEDED']

    warp_options = gdal.WarpOptions(creationOptions=create_options,
                                    multithread=True,
                                    warpMemoryLimit=8000,
                                    warpOptions=['NUM_THREADS=ALL_CPUS'],
                                    srcSRS=source_srs,
                                    dstSRS=target_srs)

    dst_ds = gdal.Warp(reprojected_geotiff_path, geotiff_ds, options=warp_options)

    return dst_ds, reprojected_geotiff_path


def correct_geotiff_projection(geotiff: str, workspace: str, target_srs='EPSG:3857') -> str:
    source_srs = find_srs(geotiff)

    if target_srs.upper() != source_srs.upper():
        return reproject_geotiff(geotiff, workspace, source_srs, target_srs)
    else:
        return geotiff


def find_geotiffs_to_reproject(geotiffs: list, workspace: str, target_srs='EPSG:3857') -> list:
    normalized_geotiffs = []

    for geotiff in geotiffs:
        normalized_geotiffs.append(correct_geotiff_projection(geotiff, workspace, target_srs))

    return normalized_geotiffs


def compress_geotiff(
        geotiff_ds: gdal.Dataset, workspace: str,
        geotiff_filename='compressed_geotiff.tif') -> [gdal.Dataset, str]:
    log_message_with_border('Running gdal_translate to compress the GeoTIFF')

    filename_and_extension = geotiff_filename.split('.')
    compressed_geotiff_path = create_tempfile_path(workspace, filename_and_extension[0], filename_and_extension[1])

    creation_options = ['COMPRESS=DEFLATE', 'TILED=YES', 'NUM_THREADS=ALL_CPUS']

    translate_options = gdal.TranslateOptions(format='GTiff', creationOptions=creation_options)

    compressed_geotiff_ds = gdal.Translate(compressed_geotiff_path, geotiff_ds, options=translate_options)

    return compressed_geotiff_ds, compressed_geotiff_path


def process_geopdf(geopdf_path: str, gdal_pdf_layers: list, workspace: str, resize_tiff: bool) -> str:
    geotiff_ds, unclipped_geotiff_path = convert_to_geotiff(geopdf_path, ','.join(gdal_pdf_layers), workspace)
    logger.info('Unclipped GeoTIFF={}'.format(unclipped_geotiff_path))

    unprojected_geojson = find_neatline(unclipped_geotiff_path)
    logger.debug(unprojected_geojson)

    srs = find_srs(unclipped_geotiff_path)
    logger.debug(srs)

    neatline_geojson_path = add_srs_to_geojson(unprojected_geojson, srs, workspace)
    logger.debug('GeoJSON with srs added={}'.format(neatline_geojson_path))

    geotiff_ds, clipped_geotiff_path = crop_geotiff(geotiff_ds, neatline_geojson_path, workspace)
    logger.debug('Cropped GeoTIFF={}'.format(clipped_geotiff_path))

    source_srs = find_srs(clipped_geotiff_path)
    geotiff_ds, reprojected_geotiff_path = reproject_geotiff(geotiff_ds, workspace, source_srs, 'EPSG:3857')

    if resize_tiff:
        geotiff_ds, resized_geotiff_path = resize_tiff_to_fit_in_tile(geotiff_ds, workspace)
        logger.debug('Resized GeoTIFF={}'.format(resized_geotiff_path))

    geotiff_ds, compressed_geotiff_path = compress_geotiff(geotiff_ds, workspace)
    logger.debug('Compressed GeoTIFF={}'.format(compressed_geotiff_path))

    close_raster_dataset(geotiff_ds)

    return compressed_geotiff_path


def generate_tiles_from_geotiff(geotiff_path: str, workspace: str, min_zoom=1, max_zoom=17, verbose=False) -> str:
    log_message_with_border('Generating TMS directory from GeoTIFF')

    if not os.path.exists(geotiff_path) or not os.path.isfile(geotiff_path):
        logger.error('{} does not exist'.format(geotiff_path))
        sys.exit(1)

    tiles_directory = os.path.join(workspace, 'tiles')

    gdal2tiles.generate_tiles(geotiff_path, tiles_directory,
                              zoom=[min_zoom, max_zoom],
                              profile='mercator',
                              s_srs=find_srs(geotiff_path),
                              resampling='lanczos',
                              nb_processes=multiprocessing.cpu_count(),
                              tmscompatible=True,
                              resume=True,
                              verbose=True)

    return tiles_directory


def resize_tiff_to_fit_in_tile(geotiff_ds: gdal.Dataset, workspace: str, block_size=256) -> [gdal.Dataset, str]:
    log_message_with_border('Using gdalwarp to resize GeoTIFF to be divisible by block size')

    resized_geotiff_path = create_tempfile_path(workspace, 'resized_geotiff', 'tif')

    width = geotiff_ds.RasterXSize
    height = geotiff_ds.RasterYSize

    new_width = int(width / block_size) * block_size
    new_height = int(height / block_size) * block_size

    create_options = ['TILED=YES', 'BIGTIFF=IF_NEEDED']

    options = gdal.WarpOptions(creationOptions=create_options,
                               multithread=True,
                               warpMemoryLimit=8000,
                               warpOptions=['NUM_THREADS=ALL_CPUS'],
                               width=new_width,
                               height=new_height,
                               format='GTiff')

    return gdal.Warp(resized_geotiff_path, geotiff_ds, options=options), resized_geotiff_path


def generate_final_file_path(name: str, output_directory: str, extension: str) -> str:
    final_mbtiles_name = '{}.{}'.format(name, extension)

    return os.path.join(output_directory, final_mbtiles_name)


def get_extension_for_format(output_format):
    if output_format == 'GTiff':
        return 'tif'
    elif output_format == 'MBTiles':
        return 'mbtiles'
    elif output_format == 'XYZ':
        return 'csv'
    elif output_format == 'TMS':
        return 'zip'


def generate_mbtiles_from_geotif(geotiff_path: str, workspace: str, verbose: bool) -> str:
    log_message_with_border('Generating MBTiles file from GeoTIFF')

    if not os.path.exists(geotiff_path) or not os.path.isfile(geotiff_path):
        logger.error('{} does not exist'.format(geotiff_path))
        sys.exit(1)

    tiles_directory = generate_tiles_from_geotiff(geotiff_path, workspace, verbose=verbose)

    mbtiles_path = create_tempfile_path(workspace, 'processed', 'mbtiles')

    log_message_with_border('Generating MBTiles file from TMS directory')

    mbutil.disk_to_mbtiles(tiles_directory, mbtiles_path, format='png', scheme='tms')

    return mbtiles_path


def do_work(args: argparse.Namespace):
    if not os.path.exists(args.output_directory) or not os.path.isdir(args.output_directory):
        logger.error('{} does not exist, please create and re-run'.format(args.output_directory))
        sys.exit(1)

    geopdf = args.geopdf

    if not os.path.exists(geopdf) or not os.path.isfile(geopdf):
        logger.error('{} does not exist or is not a file'.format(geopdf))
        sys.exit(1)

    if args.temp_directory:
        os.environ["TMPDIR"] = args.temp_directory

    if args.workspace and os.path.exists(args.workspace):
        workspace = tempfile.mkdtemp(dir=args.workspace)
    else:
        workspace = tempfile.mkdtemp()

    extension = get_extension_for_format(args.output_format)

    if args.name:
        converted_path = generate_final_file_path(args.name, args.output_directory, extension)
    else:
        name = os.path.splitext(os.path.basename(geopdf))[0]
        converted_path = generate_final_file_path(name, args.output_directory, extension)

    if not os.path.exists(converted_path) or args.overwrite:
        converted_geotiff = process_geopdf(geopdf, pdf_layers_to_include(), workspace, args.resize_tiff)

        if converted_geotiff and os.path.exists(converted_geotiff) and os.path.isfile(converted_geotiff):
            log_message_with_border('Completed')

            if args.output_format == 'MBTiles':
                converted_mbtiles = generate_mbtiles_from_geotif(converted_geotiff, workspace, args.verbose)

                if os.path.exists(converted_mbtiles) and os.path.isfile(converted_mbtiles):
                    logger.info('Moving {} to {}'.format(converted_mbtiles, converted_path))
                    shutil.move(converted_mbtiles, converted_path)
                else:
                    logger.error('{} error occured with converting GeoTIFF to MBTiles'.format(converted_mbtiles))
            elif args.output_format == 'TMS':
                tiles_directory = generate_tiles_from_geotiff(converted_geotiff, workspace)
                zipped_tiles = create_tempfile_path(workspace, 'zipped_tiles', 'zip')

                shutil.make_archive(zipped_tiles, 'zip', tiles_directory)

                logger.info('Moving {} to {}'.format(zipped_tiles, converted_path))
                shutil.move(zipped_tiles, converted_path)
            elif args.output_format == 'GTiff':
                logger.info('Moving {} to {}'.format(converted_geotiff, converted_path))
                shutil.move(converted_geotiff, converted_path)
    else:
        logger.info('{} already exists and not overritting.'.format(converted_path))

    cleanup(workspace)


if __name__ == '__main__':
    # this allows GDAL to throw Python Exceptions
    gdal.UseExceptions()

    _args = build_cmd_line_args()

    if _args.verbose:
        logging.basicConfig(level=logging.DEBUG, format="%(threadName)s::%(asctime)s::%(message)s")
    else:
        logging.basicConfig(level=logging.INFO, format="%(threadName)s::%(asctime)s::%(message)s")
    logger = logging.getLogger('convert_geopdf_to_mbtiles')

    if _args.verbose and _args.quiet:
        logger.error('Only select verbose or quiet')
        sys.exit(1)

    try:
        do_work(_args)
    except Exception:
        logger.exception('Bad stuff happened')
