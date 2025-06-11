import numpy as np
import xarray as xr
import datetime
from pyproj import Transformer

def day_lengths(
    dates: xr.DataArray,
    lat: xr.DataArray,
    obliquity: float = -0.4091,
    summer_solstice: str = "06-21",
) -> xr.DataArray:
    """
    Day-lengths according to latitude, obliquity, and day of year.
    
    Standalone implementation based on xclim.indices.generic.day_lengths
    
    Parameters
    ----------
    dates: xr.DataArray
        Time coordinate array
    lat: xr.DataArray
        Latitude coordinate in degrees
    obliquity: float
        Obliquity of the elliptic (radians). Default: -0.4091
    summer_solstice: str
        Date of summer solstice in northern hemisphere (MM-DD format). Default: "06-21"
        
    Returns
    -------
    xr.DataArray
        If start and end date provided, returns total sum of daylight-hours between dates.
        If no start and end date provided, returns day-length in hours per individual day.
    """
        
    # Get calendar
    cal = get_calendar(dates)
    
    # Calculate year length for each time point
    year_length_values = []
    for year in dates.time.dt.year.values:
        year_length_values.append(days_in_year(year, cal))
    
    year_length = dates.time.copy(data=year_length_values)
    
    # Calculate Julian date from solstice
    julian_date_from_solstice = doy_to_days_since(
        dates.time.dt.dayofyear, start=summer_solstice, calendar=cal
    )
    
    # Calculate day length using the formula from xclim
    m_lat_dayofyear = 1 - np.tan(np.radians(lat)) * np.tan(
        obliquity * (np.cos((2 * np.pi * julian_date_from_solstice) / year_length))
    )
    
    # Ensure m_lat_dayofyear is within valid range for arccos
    m_lat_dayofyear = np.clip(m_lat_dayofyear, 0, 2)
    
    day_length_hours = (np.arccos(1 - m_lat_dayofyear) / np.pi) * 24
    
    # Set attributes
    day_length_hours.attrs = {
        'units': 'hours',
        'long_name': 'Day length',
        'description': 'Length of daylight in hours'
    }
    
    return day_length_hours

def get_calendar(da):
    """Get calendar from DataArray"""
    try:
        return da.time.dt.calendar
    except AttributeError:
        return "standard"

def days_in_year(year, calendar="standard"):
    """Get number of days in a year for given calendar"""
    if calendar in ["360_day"]:
        return 360
    elif calendar in ["365_day", "noleap"]:
        return 365
    else:  # standard, gregorian, proleptic_gregorian
        return 366 if calendar.isleap(year) else 365

def doy_to_days_since(doy, start="06-21", calendar="standard"):
    """Convert day of year to days since a reference date"""
    start_month, start_day = map(int, start.split("-"))
    
    # Create reference date for each year
    # years = np.unique([d.year if hasattr(d, 'year') else d for d in doy.time.dt.year.values])
    
    result = []
    for i, (year, day_of_year) in enumerate(zip(doy.time.dt.year.values, doy.values)):
        # Get day of year for summer solstice
        if calendar == "360_day":
            solstice_doy = (start_month - 1) * 30 + start_day
        else:
            try:
                solstice_date = datetime.datetime(year, start_month, start_day)
                solstice_doy = solstice_date.timetuple().tm_yday
            except Exception as e:
                # Fallback calculation
                days_in_months = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
                if calendar.isleap(year):
                    days_in_months[1] = 29
                solstice_doy = sum(days_in_months[:start_month-1]) + start_day
        
        # Calculate days since solstice
        days_since = day_of_year - solstice_doy
        
        # Handle year wraparound
        year_length = days_in_year(year, calendar)
        if days_since < 0:
            days_since += year_length
            
        result.append(days_since)
    
    return np.array(result)

def get_lat_in_4326(da):

    if da.rio.crs is None:
        raise ValueError("The input DataArray does not have a CRS defined. Canno transform coordinate values")

    # Create transformer
    transformer = Transformer.from_crs(da.rio.crs.to_epsg(), "EPSG:4326", always_xy=True)

    # Transform coordinates
    _, lat = transformer.transform(da.x.values, da.y.values)

    return lat