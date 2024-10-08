import sqlite3


def connect_to_spatialite(db_path):
    conn = sqlite3.connect(db_path)
    conn.enable_load_extension(True)
    conn.execute("SELECT load_extension('mod_spatialite');")
    return conn


def test_spatialite(conn):
    cursor = conn.cursor()
    cursor.execute("SELECT spatialite_version();")
    version = cursor.fetchone()[0]
    print(f"SpatiaLite version: {version}")


def initialize_spatial_metadata(conn):
    cursor = conn.cursor()
    cursor.execute("SELECT InitSpatialMetaData(1);")
    conn.commit()
    cursor.close()


def list_spatialite_functions(conn):
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM sqlite_master WHERE type='function';")
    functions = cursor.fetchall()
    return functions
