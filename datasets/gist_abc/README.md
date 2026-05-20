# GIST ABC Dataset Folder

Put your images here:

```text
images/A/*.jpg
images/B/*.jpg
images/C/*.jpg
```

Expected metadata:

```csv
image_path,latitude,longitude,yaw,timestamp,building_id,split
images/A/img_0001.jpg,35.xxxxxx,126.xxxxxx,90.0,2026:05:20 12:00:00,A,train
```

`building_regions.geojson` should contain A/B/C region polygons in lon/lat coordinate order.
