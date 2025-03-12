```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import intake 
import os
import xarray as xr
import cftime
```


```python
import intake

# Load CMIP6 dataset catalog (modify if needed)
col_url = "https://storage.googleapis.com/cmip6/pangeo-cmip6.json"
col = intake.open_esm_datastore(col_url)

# Search catalog for near-surface wind speed variables
query = col.search(
    table_id="Amon",  # Monthly atmospheric variables
    variable_id=["sfcWind"],  # Near-surface wind variables
    experiment_id="historical"  # Use historical runs
)

# Convert to DataFrame
query_df = query.df  
print(query_df.head())

```

      activity_id institution_id     source_id experiment_id  member_id table_id  \
    0        CMIP      NOAA-GFDL      GFDL-CM4    historical   r1i1p1f1     Amon   
    1        CMIP           IPSL  IPSL-CM6A-LR    historical   r8i1p1f1     Amon   
    2        CMIP           IPSL  IPSL-CM6A-LR    historical   r2i1p1f1     Amon   
    3        CMIP           IPSL  IPSL-CM6A-LR    historical  r30i1p1f1     Amon   
    4        CMIP           IPSL  IPSL-CM6A-LR    historical  r31i1p1f1     Amon   
    
      variable_id grid_label                                             zstore  \
    0     sfcWind        gr1  gs://cmip6/CMIP6/CMIP/NOAA-GFDL/GFDL-CM4/histo...   
    1     sfcWind         gr  gs://cmip6/CMIP6/CMIP/IPSL/IPSL-CM6A-LR/histor...   
    2     sfcWind         gr  gs://cmip6/CMIP6/CMIP/IPSL/IPSL-CM6A-LR/histor...   
    3     sfcWind         gr  gs://cmip6/CMIP6/CMIP/IPSL/IPSL-CM6A-LR/histor...   
    4     sfcWind         gr  gs://cmip6/CMIP6/CMIP/IPSL/IPSL-CM6A-LR/histor...   
    
       dcpp_init_year   version  
    0             NaN  20180701  
    1             NaN  20180803  
    2             NaN  20180803  
    3             NaN  20180803  
    4             NaN  20180803  



```python
chosen_model = "INM-CM5-0" 

filter_model = query.df[query.df['source_id'] == chosen_model]

for _, row in filter_model.iterrows():
    member_id = row['member_id']
    dataset_url = row['zstore']

    print(f"\nEnsemble: {member_id}")

    try:
        ds = xr.open_zarr(dataset_url, consolidated=False) 
        print(f"Available variables: {list(ds.data_vars.keys())}")
    except Exception as e:
        print(f"Failed to open {member_id}: {e}")
```

    
    Ensemble: r1i1p1f1
    Available variables: ['sfcWind']
    
    Ensemble: r3i1p1f1
    Available variables: ['sfcWind']
    
    Ensemble: r4i1p1f1
    Available variables: ['sfcWind']
    
    Ensemble: r2i1p1f1



    ---------------------------------------------------------------------------

    KeyboardInterrupt                         Traceback (most recent call last)

    Cell In[48], line 13
         10 print(f"\nEnsemble: {member_id}")
         12 try:
    ---> 13     ds = xr.open_zarr(dataset_url, consolidated=False)  # Load dataset
         14     print(f"Available variables: {list(ds.data_vars.keys())}")  # Print variables
         15 except Exception as e:


    File /opt/anaconda3/envs/sea_ice_env/lib/python3.9/site-packages/xarray/backends/zarr.py:867, in open_zarr(store, group, synchronizer, chunks, decode_cf, mask_and_scale, decode_times, concat_characters, decode_coords, drop_variables, consolidated, overwrite_encoded_chunks, chunk_store, storage_options, decode_timedelta, use_cftime, zarr_version, chunked_array_type, from_array_kwargs, **kwargs)
        853     raise TypeError(
        854         "open_zarr() got unexpected keyword arguments " + ",".join(kwargs.keys())
        855     )
        857 backend_kwargs = {
        858     "synchronizer": synchronizer,
        859     "consolidated": consolidated,
       (...)
        864     "zarr_version": zarr_version,
        865 }
    --> 867 ds = open_dataset(
        868     filename_or_obj=store,
        869     group=group,
        870     decode_cf=decode_cf,
        871     mask_and_scale=mask_and_scale,
        872     decode_times=decode_times,
        873     concat_characters=concat_characters,
        874     decode_coords=decode_coords,
        875     engine="zarr",
        876     chunks=chunks,
        877     drop_variables=drop_variables,
        878     chunked_array_type=chunked_array_type,
        879     from_array_kwargs=from_array_kwargs,
        880     backend_kwargs=backend_kwargs,
        881     decode_timedelta=decode_timedelta,
        882     use_cftime=use_cftime,
        883     zarr_version=zarr_version,
        884 )
        885 return ds


    File /opt/anaconda3/envs/sea_ice_env/lib/python3.9/site-packages/xarray/backends/api.py:570, in open_dataset(filename_or_obj, engine, chunks, cache, decode_cf, mask_and_scale, decode_times, decode_timedelta, use_cftime, concat_characters, decode_coords, drop_variables, inline_array, chunked_array_type, from_array_kwargs, backend_kwargs, **kwargs)
        558 decoders = _resolve_decoders_kwargs(
        559     decode_cf,
        560     open_backend_dataset_parameters=backend.open_dataset_parameters,
       (...)
        566     decode_coords=decode_coords,
        567 )
        569 overwrite_encoded_chunks = kwargs.pop("overwrite_encoded_chunks", None)
    --> 570 backend_ds = backend.open_dataset(
        571     filename_or_obj,
        572     drop_variables=drop_variables,
        573     **decoders,
        574     **kwargs,
        575 )
        576 ds = _dataset_from_backend_dataset(
        577     backend_ds,
        578     filename_or_obj,
       (...)
        588     **kwargs,
        589 )
        590 return ds


    File /opt/anaconda3/envs/sea_ice_env/lib/python3.9/site-packages/xarray/backends/zarr.py:949, in ZarrBackendEntrypoint.open_dataset(self, filename_or_obj, mask_and_scale, decode_times, concat_characters, decode_coords, drop_variables, use_cftime, decode_timedelta, group, mode, synchronizer, consolidated, chunk_store, storage_options, stacklevel, zarr_version)
        947 store_entrypoint = StoreBackendEntrypoint()
        948 with close_on_error(store):
    --> 949     ds = store_entrypoint.open_dataset(
        950         store,
        951         mask_and_scale=mask_and_scale,
        952         decode_times=decode_times,
        953         concat_characters=concat_characters,
        954         decode_coords=decode_coords,
        955         drop_variables=drop_variables,
        956         use_cftime=use_cftime,
        957         decode_timedelta=decode_timedelta,
        958     )
        959 return ds


    File /opt/anaconda3/envs/sea_ice_env/lib/python3.9/site-packages/xarray/backends/store.py:46, in StoreBackendEntrypoint.open_dataset(self, filename_or_obj, mask_and_scale, decode_times, concat_characters, decode_coords, drop_variables, use_cftime, decode_timedelta)
         43 vars, attrs = filename_or_obj.load()
         44 encoding = filename_or_obj.get_encoding()
    ---> 46 vars, attrs, coord_names = conventions.decode_cf_variables(
         47     vars,
         48     attrs,
         49     mask_and_scale=mask_and_scale,
         50     decode_times=decode_times,
         51     concat_characters=concat_characters,
         52     decode_coords=decode_coords,
         53     drop_variables=drop_variables,
         54     use_cftime=use_cftime,
         55     decode_timedelta=decode_timedelta,
         56 )
         58 ds = Dataset(vars, attrs=attrs)
         59 ds = ds.set_coords(coord_names.intersection(vars))


    File /opt/anaconda3/envs/sea_ice_env/lib/python3.9/site-packages/xarray/conventions.py:431, in decode_cf_variables(variables, attributes, concat_characters, mask_and_scale, decode_times, decode_coords, drop_variables, use_cftime, decode_timedelta)
        424 stack_char_dim = (
        425     concat_characters
        426     and v.dtype == "S1"
        427     and v.ndim > 0
        428     and stackable(v.dims[-1])
        429 )
        430 try:
    --> 431     new_vars[k] = decode_cf_variable(
        432         k,
        433         v,
        434         concat_characters=concat_characters,
        435         mask_and_scale=mask_and_scale,
        436         decode_times=decode_times,
        437         stack_char_dim=stack_char_dim,
        438         use_cftime=use_cftime,
        439         decode_timedelta=decode_timedelta,
        440     )
        441 except Exception as e:
        442     raise type(e)(f"Failed to decode variable {k!r}: {e}")


    File /opt/anaconda3/envs/sea_ice_env/lib/python3.9/site-packages/xarray/conventions.py:281, in decode_cf_variable(name, var, concat_characters, mask_and_scale, decode_times, decode_endianness, stack_char_dim, use_cftime, decode_timedelta)
        279     var = times.CFTimedeltaCoder().decode(var, name=name)
        280 if decode_times:
    --> 281     var = times.CFDatetimeCoder(use_cftime=use_cftime).decode(var, name=name)
        283 if decode_endianness and not var.dtype.isnative:
        284     var = variables.EndianCoder().decode(var)


    File /opt/anaconda3/envs/sea_ice_env/lib/python3.9/site-packages/xarray/coding/times.py:724, in CFDatetimeCoder.decode(self, variable, name)
        722 units = pop_to(attrs, encoding, "units")
        723 calendar = pop_to(attrs, encoding, "calendar")
    --> 724 dtype = _decode_cf_datetime_dtype(data, units, calendar, self.use_cftime)
        725 transform = partial(
        726     decode_cf_datetime,
        727     units=units,
        728     calendar=calendar,
        729     use_cftime=self.use_cftime,
        730 )
        731 data = lazy_elemwise_func(data, transform, dtype)


    File /opt/anaconda3/envs/sea_ice_env/lib/python3.9/site-packages/xarray/coding/times.py:182, in _decode_cf_datetime_dtype(data, units, calendar, use_cftime)
        174 def _decode_cf_datetime_dtype(
        175     data, units: str, calendar: str, use_cftime: bool | None
        176 ) -> np.dtype:
        177     # Verify that at least the first and last date can be decoded
        178     # successfully. Otherwise, tracebacks end up swallowed by
        179     # Dataset.__repr__ when users try to view their lazily decoded array.
        180     values = indexing.ImplicitToExplicitIndexingAdapter(indexing.as_indexable(data))
        181     example_value = np.concatenate(
    --> 182         [first_n_items(values, 1) or [0], last_item(values) or [0]]
        183     )
        185     try:
        186         result = decode_cf_datetime(example_value, units, calendar, use_cftime)


    File /opt/anaconda3/envs/sea_ice_env/lib/python3.9/site-packages/xarray/core/formatting.py:77, in first_n_items(array, n_desired)
         75     indexer = _get_indexer_at_least_n_items(array.shape, n_desired, from_end=False)
         76     array = array[indexer]
    ---> 77 return np.asarray(array).flat[:n_desired]


    File /opt/anaconda3/envs/sea_ice_env/lib/python3.9/site-packages/xarray/core/indexing.py:484, in ImplicitToExplicitIndexingAdapter.__array__(self, dtype)
        483 def __array__(self, dtype: np.typing.DTypeLike = None) -> np.ndarray:
    --> 484     return np.asarray(self.get_duck_array(), dtype=dtype)


    File /opt/anaconda3/envs/sea_ice_env/lib/python3.9/site-packages/xarray/core/indexing.py:487, in ImplicitToExplicitIndexingAdapter.get_duck_array(self)
        486 def get_duck_array(self):
    --> 487     return self.array.get_duck_array()


    File /opt/anaconda3/envs/sea_ice_env/lib/python3.9/site-packages/xarray/core/indexing.py:551, in LazilyIndexedArray.get_duck_array(self)
        550 def get_duck_array(self):
    --> 551     array = self.array[self.key]
        552     # self.array[self.key] is now a numpy array when
        553     # self.array is a BackendArray subclass
        554     # and self.key is BasicIndexer((slice(None, None, None),))
        555     # so we need the explicit check for ExplicitlyIndexed
        556     if isinstance(array, ExplicitlyIndexed):


    File /opt/anaconda3/envs/sea_ice_env/lib/python3.9/site-packages/xarray/backends/zarr.py:92, in ZarrArrayWrapper.__getitem__(self, key)
         90 array = self.get_array()
         91 if isinstance(key, indexing.BasicIndexer):
    ---> 92     return array[key.tuple]
         93 elif isinstance(key, indexing.VectorizedIndexer):
         94     return array.vindex[
         95         indexing._arrayize_vectorized_indexer(key, self.shape).tuple
         96     ]


    File /opt/anaconda3/envs/sea_ice_env/lib/python3.9/site-packages/zarr/core.py:800, in Array.__getitem__(self, selection)
        798     result = self.get_orthogonal_selection(pure_selection, fields=fields)
        799 else:
    --> 800     result = self.get_basic_selection(pure_selection, fields=fields)
        801 return result


    File /opt/anaconda3/envs/sea_ice_env/lib/python3.9/site-packages/zarr/core.py:926, in Array.get_basic_selection(self, selection, out, fields)
        924     return self._get_basic_selection_zd(selection=selection, out=out, fields=fields)
        925 else:
    --> 926     return self._get_basic_selection_nd(selection=selection, out=out, fields=fields)


    File /opt/anaconda3/envs/sea_ice_env/lib/python3.9/site-packages/zarr/core.py:968, in Array._get_basic_selection_nd(self, selection, out, fields)
        962 def _get_basic_selection_nd(self, selection, out=None, fields=None):
        963     # implementation of basic selection for array with at least one dimension
        964 
        965     # setup indexer
        966     indexer = BasicIndexer(selection, self)
    --> 968     return self._get_selection(indexer=indexer, out=out, fields=fields)


    File /opt/anaconda3/envs/sea_ice_env/lib/python3.9/site-packages/zarr/core.py:1343, in Array._get_selection(self, indexer, out, fields)
       1340 if math.prod(out_shape) > 0:
       1341     # allow storage to get multiple items at once
       1342     lchunk_coords, lchunk_selection, lout_selection = zip(*indexer)
    -> 1343     self._chunk_getitems(
       1344         lchunk_coords,
       1345         lchunk_selection,
       1346         out,
       1347         lout_selection,
       1348         drop_axes=indexer.drop_axes,
       1349         fields=fields,
       1350     )
       1351 if out.shape:
       1352     return out


    File /opt/anaconda3/envs/sea_ice_env/lib/python3.9/site-packages/zarr/core.py:2179, in Array._chunk_getitems(self, lchunk_coords, lchunk_selection, out, lout_selection, drop_axes, fields)
       2177     if not isinstance(self._meta_array, np.ndarray):
       2178         contexts = ConstantMap(ckeys, constant=Context(meta_array=self._meta_array))
    -> 2179     cdatas = self.chunk_store.getitems(ckeys, contexts=contexts)
       2181 for ckey, chunk_select, out_select in zip(ckeys, lchunk_selection, lout_selection):
       2182     if ckey in cdatas:


    File /opt/anaconda3/envs/sea_ice_env/lib/python3.9/site-packages/zarr/storage.py:1426, in FSStore.getitems(self, keys, contexts)
       1422 def getitems(
       1423     self, keys: Sequence[str], *, contexts: Mapping[str, Context]
       1424 ) -> Mapping[str, Any]:
       1425     keys_transformed = {self._normalize_key(key): key for key in keys}
    -> 1426     results_transformed = self.map.getitems(list(keys_transformed), on_error="return")
       1427     results = {}
       1428     for k, v in results_transformed.items():


    File /opt/anaconda3/envs/sea_ice_env/lib/python3.9/site-packages/fsspec/mapping.py:105, in FSMap.getitems(self, keys, on_error)
        103 oe = on_error if on_error == "raise" else "return"
        104 try:
    --> 105     out = self.fs.cat(keys2, on_error=oe)
        106     if isinstance(out, bytes):
        107         out = {keys2[0]: out}


    File /opt/anaconda3/envs/sea_ice_env/lib/python3.9/site-packages/fsspec/asyn.py:118, in sync_wrapper.<locals>.wrapper(*args, **kwargs)
        115 @functools.wraps(func)
        116 def wrapper(*args, **kwargs):
        117     self = obj or args[0]
    --> 118     return sync(self.loop, func, *args, **kwargs)


    File /opt/anaconda3/envs/sea_ice_env/lib/python3.9/site-packages/fsspec/asyn.py:91, in sync(loop, func, timeout, *args, **kwargs)
         88 asyncio.run_coroutine_threadsafe(_runner(event, coro, result, timeout), loop)
         89 while True:
         90     # this loops allows thread to get interrupted
    ---> 91     if event.wait(1):
         92         break
         93     if timeout is not None:


    File /opt/anaconda3/envs/sea_ice_env/lib/python3.9/threading.py:581, in Event.wait(self, timeout)
        579 signaled = self._flag
        580 if not signaled:
    --> 581     signaled = self._cond.wait(timeout)
        582 return signaled


    File /opt/anaconda3/envs/sea_ice_env/lib/python3.9/threading.py:316, in Condition.wait(self, timeout)
        314 else:
        315     if timeout > 0:
    --> 316         gotit = waiter.acquire(True, timeout)
        317     else:
        318         gotit = waiter.acquire(False)


    KeyboardInterrupt: 



```python
ensemble_count = query.df.groupby('source_id')['member_id'].count()

ensemble_count_df = ensemble_count.reset_index()
ensemble_count_df.columns = ['Model', 'Number of Ensembles']

filter_df = ensemble_count_df[ensemble_count_df['Number of Ensembles'] >= 10]

sort_df = filter_df.sort_values(by='Number of Ensembles', ascending=False)

print(sort_df)
```

                Model  Number of Ensembles
    22      EC-Earth3                   73
    17        CanESM5                   65
    47         MIROC6                   50
    32    GISS-E2-1-G                   46
    41   IPSL-CM6A-LR                   32
    46     MIROC-ES2L                   31
    1   ACCESS-ESM1-5                   30
    14     CNRM-CM6-1                   30
    34    GISS-E2-1-H                   25
    56    UKESM1-0-LL                   19
    7           CESM2                   11
    39      INM-CM5-0                   10
    49  MPI-ESM1-2-HR                   10
    50  MPI-ESM1-2-LR                   10



```python
chosen_model = "INM-CM5-0"

filtered_ensembles = query.df[query.df['source_id'] == chosen_model][['source_id', 'member_id']]

print(filtered_ensembles)
```

         source_id  member_id
    181  INM-CM5-0   r1i1p1f1
    197  INM-CM5-0   r3i1p1f1
    198  INM-CM5-0   r4i1p1f1
    199  INM-CM5-0   r2i1p1f1
    200  INM-CM5-0   r5i1p1f1
    202  INM-CM5-0   r6i1p1f1
    203  INM-CM5-0   r7i1p1f1
    204  INM-CM5-0   r8i1p1f1
    211  INM-CM5-0   r9i1p1f1
    226  INM-CM5-0  r10i1p1f1



```python
output_dir = "wind_speed_data2"
os.makedirs(output_dir, exist_ok=True)

selected_models = filter_df['Model'].tolist()

for chosen_model in selected_models:
    print(f"Processing model: {chosen_model}")

    filter_model = query.df[(query.df['source_id'] == chosen_model) & (query.df['variable_id'] == 'sfcWind')]
    
    if filter_model.empty:
        print(f"No 'sfcWind' data.")
        continue

    wind_speed_data = {}

    for _, row in filter_model.iterrows():
        member_id = row['member_id']
        dataset_url = row['zstore']

        print(f"  Processing ensemble: {member_id}")

        try:
            ds = xr.open_zarr(dataset_url, consolidated=True)

            time = ds['time'].data
            if isinstance(time[0], cftime.datetime):
                time_mask = time >= cftime.DatetimeNoLeap(2000, 1, 1)
            else:
                time_mask = time >= np.datetime64('2000-01-01')

            ds_filtered_time = ds.sel(time=time_mask)

            if 'lat' in ds.coords:
                lat = ds['lat']
            elif 'latitude' in ds.coords:
                lat = ds['latitude']
            elif 'nav_lat' in ds.coords:
                lat = ds['nav_lat']
            else:
                print(f"    No lat variable found for {member_id}.")
                continue

            mask = (lat >= 75).compute()
            ds_north_75 = ds_filtered_time.where(mask, drop=True)

            if {'j', 'i'}.issubset(ds_north_75.dims):
                spatial_dims = ['j', 'i']
            elif {'y', 'x'}.issubset(ds_north_75.dims):
                spatial_dims = ['y', 'x']
            elif {'lat', 'lon'}.issubset(ds_north_75.dims):
                spatial_dims = ['lat', 'lon']
            else:
                print(f"    Unknown spatial dimensions for {member_id}.")
                continue

            if 'sfcWind' in ds_north_75.data_vars:
                wind_speed = ds_north_75['sfcWind'].mean(dim=spatial_dims)
                wind_speed_series = wind_speed.to_series()
                wind_speed_data[member_id] = wind_speed_series
            else:
                print(f"    'sfcWind' not found for {member_id}.")

        except Exception as e:
            print(f"    Failed to process {member_id}: {e}")

    if wind_speed_data:
        wind_speed_df = pd.DataFrame(wind_speed_data)
        csv_filename = f"{output_dir}/{chosen_model}_wind_speed.csv"
        wind_speed_df.to_csv(csv_filename, index=True)
        print(f"Saved {chosen_model}")

print("Processing complete.")
```


      Cell In[44], line 65
        print("Processed.")
                           ^
    IndentationError: unexpected unindent




```python
csv_dir = "wind_speed_data/"
csv_files = glob.glob(f"{csv_dir}*_wind_speed.csv")
```


```python
model_monthly_means = {}
model_monthly_stds = {}

for csv_file in csv_files:
    model_name = csv_file.split("/")[-1].replace("_wind_speed.csv", "")

    wind_speed_df = pd.read_csv(csv_file, index_col=0, parse_dates=True)
    
    wind_speed_monthly_mean = wind_speed_df.groupby(wind_speed_df.index.month).mean().mean(axis=1)
    wind_speed_monthly_std = wind_speed_df.groupby(wind_speed_df.index.month).std().mean(axis=1)

    model_monthly_means[model_name] = np.array(wind_speed_monthly_mean)
    model_monthly_stds[model_name] = np.array(wind_speed_monthly_std)

monthly_means_df = pd.DataFrame(model_monthly_means)
monthly_stds_df = pd.DataFrame(model_monthly_stds)

month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
month_values = np.arange(1, 13)

plt.figure(figsize=(12, 8))

for model_name in monthly_means_df.columns:
    mean_values = monthly_means_df[model_name].values
    std_values = monthly_stds_df[model_name].values  

    plt.plot(month_values, mean_values, label=model_name)
    plt.fill_between(month_values, mean_values - std_values, mean_values + std_values, alpha=0.2)

plt.title('Monthly Mean Wind Speed')
plt.xlabel('Month')
plt.ylabel('Wind Speed (m/s)')
plt.xticks(range(1, 13), month_labels)
plt.legend(title='Models', loc='upper right', fontsize=8)
plt.grid(True)
plt.show()
```


    
![png](output_7_0.png)
    



```python
era5_file = "/Users/akikomotoki/Documents/Research/sea_ice_variability/Research/71a2be00d7b74ef6244cb3220978c974.grib"

# open ERA5 GRIB file using xarray with the cfgrib engine
ds = xr.open_dataset(era5_file, engine='cfgrib')

era5_wind_speed = ds['si10']

era5_monthly = era5_wind_speed.resample(time='1M').mean()

era5_means = era5_monthly.groupby('time.month').mean(dim='time') # monthly means (average over all years for each month)
era5_avg = era5_climatology.mean(dim=['latitude', 'longitude']) # average over the spatial dimensions (latitude and longitude)

era5_monthly_means = era5_avg.values # numpy array (1D array with 12 values)

plt.figure(figsize=(12, 8))

# detrended data. plot loop from above cell
for model_name in monthly_means_df.columns:
    mean_values = monthly_means_df[model_name].values
    std_values = monthly_stds_df[model_name].values  
    plt.plot(month_values, mean_values, label=model_name)
    plt.fill_between(month_values, mean_values - std_values, mean_values + std_values, alpha=0.2)

#  ERA5 overlay
plt.plot(month_values, era5_monthly_means, 'k-o', linewidth=3, label="ERA5 Observations")

plt.title('Monthly Mean Wind Speed (Models vs ERA5 Observations)')
plt.xlabel('Month')
plt.ylabel('Wind Speed (m/s)')
plt.xticks(range(1, 13), month_labels)
plt.legend(title='Models', loc='best', fontsize=8)
plt.grid(True)
plt.show()
```

    /opt/anaconda3/envs/sea_ice_env/lib/python3.9/site-packages/xarray/core/groupby.py:508: FutureWarning: 'M' is deprecated and will be removed in a future version, please use 'ME' instead.
      index_grouper = pd.Grouper(



    
![png](output_8_1.png)
    



```python
- some differences in timing 
- EC earth (bright green) has much higher winter speed and lower summe.r more ex treme cycle
- orange ones. offset seasonal cycle. peak is earlier in season. yearly compresssed. 
- look at particular seasons. split into winter and summer.

```


```python
print("ERA5 monthly means shape:", era5_monthly_means.shape)

```

    ERA5 monthly means shape: (12,)



```python
import os
print(os.getcwd())

print(ds)
```

    /Users/akikomotoki/Documents/Research/sea_ice_variability/Research
    <xarray.Dataset>
    Dimensions:     (time: 553, latitude: 121, longitude: 1440)
    Coordinates:
        number      int64 ...
      * time        (time) datetime64[ns] 1979-01-01 1979-02-01 ... 2025-01-01
        step        timedelta64[ns] ...
        surface     float64 ...
      * latitude    (latitude) float64 90.0 89.75 89.5 89.25 ... 60.5 60.25 60.0
      * longitude   (longitude) float64 -180.0 -179.8 -179.5 ... 179.2 179.5 179.8
        valid_time  (time) datetime64[ns] ...
    Data variables:
        t2m         (time, latitude, longitude) float32 ...
        si10        (time, latitude, longitude) float32 ...
    Attributes:
        GRIB_edition:            1
        GRIB_centre:             ecmf
        GRIB_centreDescription:  European Centre for Medium-Range Weather Forecasts
        GRIB_subCentre:          0
        Conventions:             CF-1.7
        institution:             European Centre for Medium-Range Weather Forecasts
        history:                 2025-03-02T13:20 GRIB to CDM+CF via cfgrib-0.9.1...



```python
all_ensemble_means = []

for csv_file in csv_files:
  
    wind_speed_df = pd.read_csv(csv_file, index_col=0, parse_dates=True)
    wind_speed_monthly_mean = wind_speed_df.groupby(wind_speed_df.index.month).mean().mean(axis=1)
    all_ensemble_means.append(wind_speed_monthly_mean.to_numpy())
    ensemble_means_array = np.array(all_ensemble_means)

ensemble_mean = np.mean(ensemble_means_array, axis=0)
ensemble_median = np.median(ensemble_means_array, axis=0)
percentile_25 = np.percentile(ensemble_means_array, 25, axis=0)
percentile_75 = np.percentile(ensemble_means_array, 75, axis=0)
percentile_5 = np.percentile(ensemble_means_array, 5, axis=0)
percentile_95 = np.percentile(ensemble_means_array, 95, axis=0)

month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
# month_values = np.arange(1, 13)

plt.figure(figsize=(12, 8))

# shaded 25th and 75th percentile
plt.fill_between(month_values, percentile_25, percentile_75, color='purple', alpha=0.3, label='25th-75th Percentile Range')

# mean & median
plt.plot(month_values, ensemble_mean, color='purple', linestyle='--', linewidth=2, label='Ensemble Mean')
plt.plot(month_values, ensemble_median, color='purple', linestyle='-', linewidth=2, label='Ensemble Median')

# 5th and 95th percentiles
plt.plot(month_values, percentile_5, color='purple', linestyle=':', linewidth=1.5, label='5th Percentile')
plt.plot(month_values, percentile_95, color='purple', linestyle=':', linewidth=1.5, label='95th Percentile')

#  ERA5 overlay
plt.plot(month_values, era5_monthly_means, 'k-o', linewidth=3, label="ERA5 Observations")

plt.title('Monthly Mean Wind Speed Across CMIP6 Models')
plt.xlabel('Month')
plt.ylabel('Wind Speed (m/s)')
plt.xticks(range(1, 13), month_labels)
plt.legend(title='Statistics', loc='lower left', fontsize=8)
plt.grid(True)
plt.show()
```


    
![png](output_12_0.png)
    


intermodel spread. add median. large outliers. skewed. 


```python
from scipy.signal import detrend
from scipy.optimize import curve_fit

def sinusoidal_fit(t, A, phi, C):
    return A * np.cos(2 * np.pi * t / 12 + phi) + C

detrended_data_dict = {}
variance_dict = {}

for csv_file in csv_files:
    model_name = csv_file.split("/")[-1].replace("_wind_speed.csv", "")

    wind_speed_df = pd.read_csv(csv_file, index_col=0, parse_dates=True)
    wind_speed_df.index = pd.to_datetime(wind_speed_df.index)
    wind_speed_monthly_mean = wind_speed_df.groupby(wind_speed_df.index.month).mean().mean(axis=1)

    detrended_values = detrend(wind_speed_monthly_mean.to_numpy(), type='linear')

    variance = np.var(detrended_values)
    std_dev = np.sqrt(variance)

    detrended_data_dict[model_name] = detrended_values
    variance_dict[model_name] = std_dev

detrended_df = pd.DataFrame(detrended_data_dict)

# mean and standard deviation across all models
mean_detrended = detrended_df.mean(axis=1)
std_detrended = detrended_df.std(axis=1)

months = np.arange(1, 13)

# sinusoidal function fit to mean detrended wind speed
popt, _ = curve_fit(sinusoidal_fit, months, mean_detrended, p0=[1, 0, 0])
A_fit, phi_fit, C_fit = popt

fitted_curve = sinusoidal_fit(months, *popt)

# timing of minimum wind speed (convert phase shift to months)
min_month = (12 * (-phi_fit / (2 * np.pi))) % 12

fit_results = {
    "Amplitude": A_fit,
    "Phase Offset (rad)": phi_fit,
    "Mean Offset": C_fit,
    "Minimum Wind Month": min_month
}
fit_results_df = pd.DataFrame([fit_results])

plt.figure(figsize=(12, 8))

# plot all model detrended
for model_name in detrended_df.columns:
    mean_values = detrended_df[model_name].to_numpy().flatten()
    std_dev = variance_dict[model_name]
    plt.plot(month_values, mean_values, label=model_name)
    plt.fill_between(month_values, mean_values - std_dev, mean_values + std_dev, alpha=0.2)

# plot best fit sinusoidal
plt.plot(months, fitted_curve, '--k', linewidth=2, label="Sinusoidal Fit (All Models)")

plt.title('Detrended Monthly Wind Speed')
plt.xlabel('Month')
plt.ylabel('Detrended Wind Speed (m/s)')
plt.xticks(range(1, 13), month_labels)
plt.legend(title='Models', loc='lower right', fontsize=8)
plt.grid(True)
plt.show()

from IPython.display import display
display(fit_results_df)
```

    /var/folders/t7/4flnfkx542q3fx0syy1n_pww0000gn/T/ipykernel_60393/3160072285.py:34: OptimizeWarning: Covariance of the parameters could not be estimated
      popt, _ = curve_fit(sinusoidal_fit, months, mean_detrended, p0=[1, 0, 0])



    
![png](output_14_1.png)
    



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Amplitude</th>
      <th>Phase Offset (rad)</th>
      <th>Mean Offset</th>
      <th>Minimum Wind Month</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.512938</td>
      <td>-0.227396</td>
      <td>1.076936e-10</td>
      <td>0.434294</td>
    </tr>
  </tbody>
</table>
</div>



```python
offsets = {}

for model_name in detrended_df.columns:
    
    model_signal = detrended_df[model_name].to_numpy().flatten()
    best_corr = -np.inf
    best_shift = None
    
    # shifting reference sinusoid by integer months from -6 to +6
    for shift in range(-6, 7):
        shifted_reference = np.roll(fitted_curve, shift)
        
        # Pearson correlation between the model's signal and the shifted reference
        corr = np.corrcoef(model_signal, shifted_reference)[0, 1]
        if corr > best_corr:
            best_corr = corr
            best_shift = shift

    offsets[model_name] = best_shift

offsets_df = pd.DataFrame.from_dict(offsets, orient="index", columns=["Phase Offset (months)"])
print("Phase Offsets Relative to the Reference Sinusoid:")

from IPython.display import display
display(offsets_df)

```

    Phase Offsets Relative to the Reference Sinusoid:



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Phase Offset (months)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>ACCESS-ESM1-5</th>
      <td>0</td>
    </tr>
    <tr>
      <th>IPSL-CM6A-LR</th>
      <td>-1</td>
    </tr>
    <tr>
      <th>MIROC-ES2L</th>
      <td>0</td>
    </tr>
    <tr>
      <th>INM-CM5-0</th>
      <td>0</td>
    </tr>
    <tr>
      <th>MPI-ESM1-2-LR</th>
      <td>0</td>
    </tr>
    <tr>
      <th>CNRM-CM6-1</th>
      <td>0</td>
    </tr>
    <tr>
      <th>MIROC6</th>
      <td>0</td>
    </tr>
    <tr>
      <th>CESM2</th>
      <td>0</td>
    </tr>
    <tr>
      <th>CanESM5</th>
      <td>0</td>
    </tr>
    <tr>
      <th>GISS-E2-1-H</th>
      <td>0</td>
    </tr>
    <tr>
      <th>MPI-ESM1-2-HR</th>
      <td>0</td>
    </tr>
    <tr>
      <th>GISS-E2-1-G</th>
      <td>0</td>
    </tr>
    <tr>
      <th>EC-Earth3</th>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



```python
# range of fractional shifts to test (-6 to +6 in 0.1 month increments)
shifts = np.arange(-6, 6.01, 0.1)

offsets = {}

for model in detrended_df.columns:
   
    model_data = detrended_df[model].to_numpy().flatten()
    best_corr = -np.inf
    best_shift = None
    
    # each fractional shift
    for shift in shifts:
        candidate = sinusoidal_fit(months + shift, A_fit, phi_fit, C_fit)
        corr = np.corrcoef(model_data, candidate)[0, 1]
        if corr > best_corr:
            best_corr = corr
            best_shift = shift
            
    offsets[model] = best_shift

offsets_df = pd.DataFrame.from_dict(offsets, orient="index", columns=["Phase Offset (months)"])
offsets_df = offsets_df.round(2)

print("Fractional Phase Offsets Relative to the Reference Sinusoid:")

from IPython.display import display
display(offsets_df)

```

    Fractional Phase Offsets Relative to the Reference Sinusoid:



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Phase Offset (months)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>ACCESS-ESM1-5</th>
      <td>-0.1</td>
    </tr>
    <tr>
      <th>IPSL-CM6A-LR</th>
      <td>1.0</td>
    </tr>
    <tr>
      <th>MIROC-ES2L</th>
      <td>-0.1</td>
    </tr>
    <tr>
      <th>INM-CM5-0</th>
      <td>-0.3</td>
    </tr>
    <tr>
      <th>MPI-ESM1-2-LR</th>
      <td>-0.0</td>
    </tr>
    <tr>
      <th>CNRM-CM6-1</th>
      <td>0.2</td>
    </tr>
    <tr>
      <th>MIROC6</th>
      <td>0.1</td>
    </tr>
    <tr>
      <th>CESM2</th>
      <td>-0.0</td>
    </tr>
    <tr>
      <th>CanESM5</th>
      <td>-0.1</td>
    </tr>
    <tr>
      <th>GISS-E2-1-H</th>
      <td>-0.3</td>
    </tr>
    <tr>
      <th>MPI-ESM1-2-HR</th>
      <td>0.1</td>
    </tr>
    <tr>
      <th>GISS-E2-1-G</th>
      <td>-0.2</td>
    </tr>
    <tr>
      <th>EC-Earth3</th>
      <td>-0.0</td>
    </tr>
  </tbody>
</table>
</div>



```python
# Initialize a dictionary to store the MSE for each model
mse_dict = {}

# Loop through each model in the detrended DataFrame
for model in detrended_df.columns:
    # Convert the model's detrended monthly values to a 1D numpy array
    model_data = detrended_df[model].to_numpy().flatten()
    
    # Get the best fractional shift for this model (assume this dictionary exists)
    best_shift = offsets[model]  # e.g., computed from earlier code
    
    # Evaluate the reference sinusoid at shifted month values
    fitted_model = sinusoidal_fit(months + best_shift, A_fit, phi_fit, C_fit)
    
    # Compute the Mean Square Error between the model data and its shifted sinusoid fit
    mse = np.mean((model_data - fitted_model)**2)
    mse_dict[model] = mse

# Convert the MSE dictionary to a DataFrame for display
mse_df = pd.DataFrame.from_dict(mse_dict, orient="index", columns=["MSE"])

print("Mean Squared Error (MSE) of the Sinusoidal Fit for Each Model:")

from IPython.display import display

combined_df = offsets_df.join(mse_df)
combined_df = combined_df.reset_index().rename(columns={'index': 'Model'})
display(combined_df)

```

    Mean Squared Error (MSE) of the Sinusoidal Fit for Each Model:



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Model</th>
      <th>Phase Offset (months)</th>
      <th>MSE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ACCESS-ESM1-5</td>
      <td>-0.1</td>
      <td>0.026410</td>
    </tr>
    <tr>
      <th>1</th>
      <td>IPSL-CM6A-LR</td>
      <td>1.0</td>
      <td>0.058569</td>
    </tr>
    <tr>
      <th>2</th>
      <td>MIROC-ES2L</td>
      <td>-0.1</td>
      <td>0.104163</td>
    </tr>
    <tr>
      <th>3</th>
      <td>INM-CM5-0</td>
      <td>-0.3</td>
      <td>0.016251</td>
    </tr>
    <tr>
      <th>4</th>
      <td>MPI-ESM1-2-LR</td>
      <td>-0.0</td>
      <td>0.042097</td>
    </tr>
    <tr>
      <th>5</th>
      <td>CNRM-CM6-1</td>
      <td>0.2</td>
      <td>0.037250</td>
    </tr>
    <tr>
      <th>6</th>
      <td>MIROC6</td>
      <td>0.1</td>
      <td>0.017952</td>
    </tr>
    <tr>
      <th>7</th>
      <td>CESM2</td>
      <td>-0.0</td>
      <td>0.095013</td>
    </tr>
    <tr>
      <th>8</th>
      <td>CanESM5</td>
      <td>-0.1</td>
      <td>0.020237</td>
    </tr>
    <tr>
      <th>9</th>
      <td>GISS-E2-1-H</td>
      <td>-0.3</td>
      <td>0.018679</td>
    </tr>
    <tr>
      <th>10</th>
      <td>MPI-ESM1-2-HR</td>
      <td>0.1</td>
      <td>0.018818</td>
    </tr>
    <tr>
      <th>11</th>
      <td>GISS-E2-1-G</td>
      <td>-0.2</td>
      <td>0.023249</td>
    </tr>
    <tr>
      <th>12</th>
      <td>EC-Earth3</td>
      <td>-0.0</td>
      <td>0.018110</td>
    </tr>
  </tbody>
</table>
</div>


- look at mean squre error. take full timme series on both and look at mean square difference. 
