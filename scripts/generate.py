from PIL import Image
from codecarbon import EmissionsTracker

tracker = EmissionsTracker(
    project_name="uvmoondream",
    output_dir="emissions",
    #country_iso_code="FR",
)

tracker.start()

def generate(model, image, task):
    # Load your image
    image_path = "images" + "/" + "page0000" + image + ".jpg"
    image = Image.open(image_path)

    # Optionally set sampling settings
    settings = {"temperature": 0.5, "max_tokens": 768, "top_p": 0.3}

    if task == "caption":
        # Generate a caption
        result = model.caption(
            image,
            length="long",
            settings=settings,
        ) # type: ignore
        emissions = tracker.stop()
        print(f"Emissions: {emissions} kg CO2")
        return result
    
    elif task == "query":
        # Answer a query
        result = model.query(
            image,
            "Your tasks: separately describe what is happening in each panel.",
            settings=settings
        ) # type: ignore
        return result
    elif task == "reasonquery":
        # Answer a query with reasoning
        result = model.query(
            image, "Your tasks: separately describe what is happening in each panel.", settings=settings, reasoning=True
        ) # type: ignore
        return result
    
    #missions = tracker.stop()
    #emissions = tracker.stop()
    
    
    
    
