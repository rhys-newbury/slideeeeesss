from PIL import Image

# List your image filenames in order
frames = [Image.open(f"/home/taco/Downloads/{i}.drawio.png") for i in range(1, 6)]

# Save as GIF
frames[0].save(
    "output.gif",
    save_all=True,
    append_images=frames[1:],
    duration=1000,   # time per frame in ms
    loop=2          # 0 = infinite loop
)
