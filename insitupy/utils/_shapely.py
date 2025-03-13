from shapely import affinity, wkt


def scale_wkt_polygon(polygon_wkt, constant):
    """
        Scales the polygon by a given constant.

        Args:
            polygon_wkt (str): Polygon in WKT format.
            constant (float): Scaling constant.

        Returns:
            Polygon: Scaled polygon.
    """
    polygon = wkt.loads(polygon_wkt)
    divided_polygon = affinity.scale(polygon, xfact=1/constant, yfact=1/constant, origin=(0, 0))
    return divided_polygon