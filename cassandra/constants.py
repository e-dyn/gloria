"""
Constant definitions used throughout the Cassandra code
"""

### --- Module Imports --- ###
# Standard Library

# Third Party

# Inhouse Packages

### --- Global Constants Definitions --- ###



### --- Class and Function Definitions --- ###

# Conversion factors to seconds. The keys correspond to pd.Timedelta units
_T_CONVERSION = {
    's': 1,
    'min': 60,
    'h': 60*60,
    'd': 24*60*60,
    'W': 7*24*60*60,
}

## Reserved names ##
# The delimiter is mainly used to construct feature matrix column names
_DELIM = '__delim__'
# Column name for the timestamp column converted to integer values
_T_INT = 'ds_int'
