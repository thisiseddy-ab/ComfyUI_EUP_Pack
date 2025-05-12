class TensorService(): 

    def getSlice(self, tensor, tileSize):
        """
        Extract a slice from the input tensor at the specified location and size.
        
        Args:
            tensor: The input tensor (e.g., a latent tensor).
            tileSize: A tuple containing (x, y, tile_w, tile_h), where
                - x: The starting width (x-coordinate) of the slice.
                - y: The starting height (y-coordinate) of the slice.
                - tile_w: The width length of the slice.
                - tile_h: The height length of the slice.
        
        Returns:
            A slice of the input tensor.
        """
        x, y, tile_w, tile_h = tileSize
        
        # Extract the slice using narrow() for the height and width dimensions
        t = tensor.narrow(-2, y, tile_h)  # Narrow along the height (second-to-last dimension)
        t = t.narrow(-1, x, tile_w)       # Narrow along the width (last dimension)
        return t
    
    
    def getTensorfromLatentImage(self, latentImage):
        """
        Extract the tensor from the latent image.
        
        Args:
            latentImage: The input latent image.
        
        Returns:
            The tensor extracted from the latent image.
        """
        #### Tile Shape ####
        return latentImage.get("samples")
    
    def setTensorInLatentImage(self, latentImage, newSamples):
        """
        Replace the 'samples' tensor in the latent image dictionary.

        Args:
            latentImage (dict): The latent image dictionary.
            new_samples (torch.Tensor): The new tensor to set as 'samples'.

        Returns:
            dict: A modified copy of the latent image with updated 'samples'.
        """

        latentImage.update({"samples": newSamples})
        return latentImage
    
    def getShapefromLatentImage(self, latentImage):
        latentTensor = self.getTensorfromLatentImage(latentImage)
        return latentTensor.shape
    
    def getNoiseMaskfromLatentImage(self, latentImage):
        return latentImage.get("noise_mask")
    
    def getBatchIndexfromLatentTensor(self, latentImage):
        return latentImage.get("batch_index", 0) 
