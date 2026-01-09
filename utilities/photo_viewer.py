import webbrowser

class PhotoViewer:
    def open_urls(self, urls: list[str]):
        for url in urls:
            webbrowser.open(url)

    def show_photos(self, photos: list[dict], url_generator):
        """
        Generate presigned URLs and open photos in browser.
        """
        urls = []

        for photo in photos:
            bucket = photo.get("bucket")
            key = photo.get("photo_key")

            if bucket and key:
                url = url_generator.generate_presigned_url(bucket, key)
                if url:
                    urls.append(url)

        if urls:
            self.open_urls(urls)
        else:
            print("Could not generate URLs for photos")

