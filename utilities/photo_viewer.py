import webbrowser

class PhotoViewer:
    """Display photo URLs in CLI (EC2-friendly)."""
    
    def show_photos(self, photos: list[dict], url_generator):
        """
        Generate presigned URLs and display them in terminal.
        
        Args:
            photos: List of photo dicts with 'bucket' and 'photo_key'
            url_generator: S3PhotoResolver instance
        """
        if not photos:
            print("\nNo photos to display")
            return
        
        print(f"\n{'='*70}")
        print(f"  📷 Found {len(photos)} Photo(s)")
        print(f"{'='*70}\n")
        
        success_count = 0
        
        for i, photo in enumerate(photos, 1):
            bucket = photo.get("bucket")
            photo_key = photo.get("photo_key")
            photo_id = photo.get("photo_id", "unknown")
            taken_at = photo.get("taken_at", "unknown date")
            
            if not bucket or not photo_key:
                print(f"{i}. ✗ Missing bucket or photo_key")
                continue
            
            # Generate presigned URL
            url = url_generator.generate_presigned_url(bucket, photo_key)
            
            if url:
                success_count += 1
                print(f"{i}. Photo ID: {photo_id[:16]}...")
                print(f"   Taken: {taken_at}")
                print(f"   URL: \033[4;36m{url}\033[0m\n")  # Blue underlined (clickable in many terminals)
            else:
                print(f"{i}. ✗ Failed to generate URL for {photo_id[:16]}...\n")
        
        if success_count > 0:
            print(f"{'='*70}")
            print(f"  ✓ Generated {success_count} URL(s)")
            print(f"  💡 Tip: Ctrl+Click URLs to open in browser")
            print(f"{'='*70}\n")
        else:
            print("✗ Could not generate any URLs\n")


