
The workflow, requires a job that is being ran, for instance:
 
```julia
jobs = [Dict()]
push!(jobs[1], "layer_name" => "functionality")
push!(jobs[1], "function" => "ConScape.connected_habitat")
 
kwargs = Dict()
push!(kwargs, :connectivity_function => ConScape.expected_cost)
push!(kwargs, :distance_transformation => x -> exp(-x/75))
 
push!(jobs[1], "kwargs" => kwargs)
 
push!(jobs, Dict())
 
push!(jobs[2], "layer_name" => "flow")
push!(jobs[2], "function" => "ConScape.betweenness_kweighted")
push!(jobs[2], "kwargs" => kwargs)
 
theta_exp = collect(-1:-2:-4)
thetas = map(x -> 10.0^x, theta_exp)
 
thetas_jobs = [
    Dict(
        "theta" => thetas[1],
        "jobs" => jobs,
        "file_name" => "RSP_" * string(thetas[1]) * ".tif"
        ),
    Dict(
        "theta" => thetas[2],
        "jobs" => jobs,
        "file_name" => "RSP_" * string(thetas[2])  * ".tif"
        )
    ]
 
 
print(Term.Tree(thetas_jobs))
```
 
It also takes a named raster stack:
 
```julia
rast = RasterStack((; zip([:source_qualities, :target_qualities, :affinities], [rast, rast, rast])...))
```
I don’t remember if all layers are required, I think you can just provide a :qualities layer
 
As I mentioned the moving window computation requires a center size for the moving window, in this case 10x10:

```julia
cntr_size = 10
radius = 500
```
 
There is also radius, I think it is in the same units as your map coordinates (but I am not 100% sure)
 
You can then run a moving windowed computation:
 
```julia
res = moving_window_conscape(rast, radius, thetas_jobs, cntr_size, ConScape.MinusLog());
```
 
However, I have found that this takes a very long time, and I have lost a lot of computations due to server restarts or crashes.
Hence, I added to run the computation by tiles, and save the intermediate results. I tried to make the tiles the size of what can computed in a 1-2 days on our server.
Herefore, I create first a dataframe with the tiles, which I can use to resume the computation.
 
```julia
tile_size=50
overlap = Int64(radius/resolution(rast)[1])
 
tiles_tab = make_tiles_table(rast, tile_size, overlap)
```
 
Which can now be run:
```julia
outdir = "C:/ tmp/"
tiled_conscape(tiles_tab, outdir, rast, radius, thetas_jobs, cntr_size, ConScape.MinusLog())
```
 
From the data directory we can "stich_tiles" back to a full raster:
```julia
stitch_conscape_tiles(outdir, rast)
cleanup_conscape_tiles(outdir)
```
 
There is also a function to map the progress, while it is running:
```julia
tile_progress_map(tiles_tab, outdir)
```
 