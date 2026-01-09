import webbrowser

class PhotoViewer:
    """CLI helper to open photos in browser."""

    def open_urls(self, urls: list[str]):
        if not urls:
            print("No photos to show")
            return

        print(f"\nOpening {len(urls)} photo(s)...\n")
        for url in urls:
            webbrowser.open(url)
