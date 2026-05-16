from typing import List, Dict, Any
# from app.core.config import settings # Uncomment if using your settings

class ChunkingService:
    def __init__(
        self,
        chunk_size: int = 500, # settings.CHUNK_SIZE
        chunk_overlap: int = 50, # settings.CHUNK_OVERLAP
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # The order of separators matters: from largest logical break to smallest
        self.separators = ["\n\n", "\n", ".", "،", "؟", " ", ""]

    def _recursive_split(self, text: str, separators: List[str]) -> List[str]:
        """
        Recursively splits the text using the provided separators.
        """
        # Base Case: If the text is already small enough, return it.
        if len(text) <= self.chunk_size:
            return [text]

        # Find the appropriate separator from our list
        active_separator = separators[-1] # Default to empty string (character level)
        next_separators = []

        for i, sep in enumerate(separators):
            if sep == "":
                active_separator = sep
                break
            if sep in text:
                active_separator = sep
                next_separators = separators[i + 1:]
                break

        # Split the text
        if active_separator:
            raw_splits = text.split(active_separator)
        else:
            raw_splits = list(text) # Character by character fallback

        # Process the splits
        valid_splits = []
        for split in raw_splits:
            if len(split) <= self.chunk_size:
                valid_splits.append(split)
            else:
                # If a split is still too large, recurse with the remaining separators
                if next_separators:
                    valid_splits.extend(self._recursive_split(split, next_separators))
                else:
                    # Absolute fallback: cut it hard at chunk_size
                    for i in range(0, len(split), self.chunk_size):
                        valid_splits.append(split[i:i+self.chunk_size])

        # Merge the small valid splits into chunks honoring the overlap
        return self._merge_splits(valid_splits, active_separator)

    def _merge_splits(self, splits: List[str], separator: str) -> List[str]:
        """
        Merges smaller splits together until the chunk_size is reached, 
        then applies the chunk_overlap logic.
        """
        chunks = []
        current_chunk = []
        current_length = 0

        for split in splits:
            split_length = len(split)
            separator_length = len(separator) if current_chunk else 0
            
            # If adding this split exceeds our chunk_size limits
            if current_length + split_length + separator_length > self.chunk_size and current_chunk:
                # 1. Save the current chunk
                chunks.append(separator.join(current_chunk))
                
                # 2. Handle overlap for the next chunk
                # Keep removing elements from the start of current_chunk until 
                # the remaining length is within our allowed overlap
                while current_length > self.chunk_overlap and current_chunk:
                    popped_element = current_chunk.pop(0)
                    current_length -= len(popped_element) + (len(separator) if current_chunk else 0)

            # Add the current split to our running chunk
            current_chunk.append(split)
            current_length += split_length + (len(separator) if len(current_chunk) > 1 else 0)

        # Add any remaining text as the last chunk
        if current_chunk:
            chunks.append(separator.join(current_chunk))

        return chunks

    def chunk_document(self, doc: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Split a parsed document dict into overlap-aware chunks with metadata."""
        text = doc.get("text", "")
        if not text.strip():
            return []

        splits = self._recursive_split(text, self.separators)
        chunks = []
        
        for idx, chunk_text in enumerate(splits):
            if not chunk_text.strip():
                continue
                
            cid = f"{doc.get('stem', 'doc')}_c{idx:04d}"
            chunks.append({
                "chunk_id": cid,
                "text": chunk_text.strip(),
                "metadata": {
                    "filename": doc.get("filename", "unknown"),
                    "stem": doc.get("stem", "unknown"),
                    "source": doc.get("source", "unknown"),
                    "chunk_id": cid,
                    "chunk_index": idx,
                    "total_chunks": len(splits),
                    "language": doc.get("language", "unknown"),
                    "arabic_ratio": doc.get("arabic_ratio", 0.0),
                    "pages": doc.get("pages", 0),
                },
            })
        return chunks

    def chunk_documents(self, docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        all_chunks = []
        for doc in docs:
            ch = self.chunk_document(doc)
            all_chunks.extend(ch)
            print(f"[Chunker] 📄 {doc.get('filename', 'Unknown')} → {len(ch)} chunks")
        print(f"[Chunker] Total chunks: {len(all_chunks)}")
        return all_chunks