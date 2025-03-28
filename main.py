import get_data.retrieve_data as get_data

get_data_url = "https://www.ncei.noaa.gov/cdo-web/api/v2/data?datasetid=GSOM"
data_categories_url = "https://www.ncei.noaa.gov/cdo-web/api/v2/datacategories/?datasetid=GSOM"
# output the api request from NOAA
# get_data.send_request()
# get_data.send_request(url=data_categories_url)
get_data.get_dataset(url=get_data_url)


