"""Schema auto-discovery via spectral clustering on SPLADE co-activation matrix.

The SchemaDiscovery class derives an AnchorSchema from a corpus without manual
anchor definitions, using spectral clustering on the co-activation matrix built
from SPLADE token activations.

Algorithm:
1. Build co-activation matrix from SPLADE encodings (NPMI)
2. Filter to top-F frequent tokens (default F=2000)
3. Compute normalized graph Laplacian
4. Extract bottom-K eigenvectors and run k-means clustering
5. Label clusters and infer anchor types
6. For code mode, replace relation clusters with eight built-in anchors
"""

import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.cluster.vq import kmeans2
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh

from infon.encoder import SpladeEncoder
from infon.schema import CODE_RELATION_ANCHORS, Anchor, AnchorSchema


class SchemaDiscovery:
    """Auto-discover AnchorSchema from a corpus via spectral clustering.
    
    Uses SPLADE encoder to build a co-activation matrix, then applies spectral
    clustering to derive anchor clusters. For code mode, replaces relation
    clusters with the eight built-in code relation anchors.
    """

    def __init__(
        self,
        n_clusters: int = 50,
        top_tokens: int = 2000,
        min_activation: float = 0.1,
    ):
        """Initialize schema discovery.
        
        Args:
            n_clusters: Number of anchor clusters to discover (default 50)
            top_tokens: Number of most frequent tokens to retain (default 2000)
            min_activation: Minimum SPLADE activation threshold (default 0.1)
        """
        self.n_clusters = n_clusters
        self.top_tokens = top_tokens
        self.min_activation = min_activation
        self.encoder = SpladeEncoder()

    def discover(self, corpus_path: str, mode: str = "code") -> AnchorSchema:
        """Discover AnchorSchema from a corpus.
        
        Args:
            corpus_path: Path to directory containing corpus files
            mode: 'code' or 'text' (determines anchor type inference)
            
        Returns:
            AnchorSchema with auto-discovered anchors
            
        Raises:
            ValueError: If mode is not 'code' or 'text'
        """
        if mode not in ("code", "text"):
            raise ValueError(f"mode must be 'code' or 'text', got: {mode}")
        
        corpus_path = Path(corpus_path)
        
        # Collect all source files
        if mode == "code":
            source_files = self._collect_code_files(corpus_path)
        else:
            source_files = self._collect_text_files(corpus_path)
        
        # Emit warning if corpus is small
        if len(source_files) < 50:
            warnings.warn(
                f"Corpus has fewer than 50 files ({len(source_files)} found); "
                "auto-discovered schema may be noisy. Consider providing a manual "
                "schema with --schema.",
                UserWarning,
            )
        
        # Build co-activation matrix from corpus
        coactivation_matrix, token_ids, token_freq = self._build_coactivation_matrix(
            source_files
        )
        
        # Filter to top frequent tokens
        filtered_matrix, filtered_token_ids = self._filter_frequent_tokens(
            coactivation_matrix, token_ids, token_freq
        )
        
        # Run spectral clustering
        clusters = self._spectral_cluster(filtered_matrix)
        
        # Build anchors from clusters
        anchors = self._build_anchors_from_clusters(
            clusters, filtered_token_ids, token_freq, mode
        )
        
        # For code mode, ensure eight built-in relation anchors are present
        if mode == "code":
            for key, anchor in CODE_RELATION_ANCHORS.items():
                if key not in anchors:
                    anchors[key] = anchor
        
        return AnchorSchema(
            anchors=anchors,
            version="auto-1.0",
            language=mode,
        )

    def _collect_code_files(self, root: Path) -> list[Path]:
        """Collect all code files from directory tree.
        
        Args:
            root: Root directory to search
            
        Returns:
            List of paths to code files (.py, .ts, .tsx, .js, .jsx)
        """
        extensions = {".py", ".ts", ".tsx", ".js", ".jsx"}
        files = []
        
        for ext in extensions:
            files.extend(root.rglob(f"*{ext}"))
        
        # Filter out __pycache__ and other common exclusions
        return [
            f for f in files
            if "__pycache__" not in f.parts and "node_modules" not in f.parts
        ]

    def _collect_text_files(self, root: Path) -> list[Path]:
        """Collect all text files from directory tree.
        
        Args:
            root: Root directory to search
            
        Returns:
            List of paths to text files (.txt, .md)
        """
        extensions = {".txt", ".md"}
        files = []
        
        for ext in extensions:
            files.extend(root.rglob(f"*{ext}"))
        
        return files

    def _build_coactivation_matrix(
        self, source_files: list[Path]
    ) -> tuple[dict[tuple[int, int], float], list[int], dict[int, int]]:
        """Build co-activation matrix from SPLADE encodings.
        
        For each pair of tokens that co-activate in the same text unit,
        accumulates their positive pointwise mutual information (NPMI).
        
        Args:
            source_files: List of source file paths to process
            
        Returns:
            Tuple of (coactivation_matrix, token_ids, token_frequencies)
            where coactivation_matrix is a dict mapping (token_id, token_id) -> npmi_score
        """
        # Track token occurrences and co-occurrences
        token_freq: dict[int, int] = defaultdict(int)
        cooccur: dict[tuple[int, int], int] = defaultdict(int)
        total_units = 0
        
        # Process each file
        for file_path in source_files:
            try:
                # Read file content
                text = file_path.read_text(encoding="utf-8", errors="ignore")
                
                # Skip empty files
                if not text.strip():
                    continue
                
                # Split into units (lines for code, sentences for text)
                # For simplicity, use lines for both modes
                units = [line.strip() for line in text.split("\n") if line.strip()]
                
                for unit in units:
                    # Encode unit
                    sparse_vec = self.encoder.encode_sparse(unit)
                    
                    # Filter by activation threshold
                    active_tokens = [
                        tid for tid, val in sparse_vec.items()
                        if val >= self.min_activation
                    ]
                    
                    if not active_tokens:
                        continue
                    
                    total_units += 1
                    
                    # Update token frequencies
                    for tid in active_tokens:
                        token_freq[tid] += 1
                    
                    # Update co-occurrence counts
                    for i, tid1 in enumerate(active_tokens):
                        for tid2 in active_tokens[i:]:  # Include self-pairs
                            pair = tuple(sorted([tid1, tid2]))
                            cooccur[pair] += 1
                            
            except Exception as e:
                # Skip files that fail to parse
                warnings.warn(f"Failed to process {file_path}: {e}")
                continue
        
        # Build NPMI matrix
        coactivation_matrix: dict[tuple[int, int], float] = {}
        
        if total_units == 0:
            # Empty corpus - return empty matrix
            return coactivation_matrix, [], token_freq
        
        for (tid1, tid2), cooccur_count in cooccur.items():
            if tid1 not in token_freq or tid2 not in token_freq:
                continue
            
            # Calculate NPMI (Normalized Pointwise Mutual Information)
            p_tid1 = token_freq[tid1] / total_units
            p_tid2 = token_freq[tid2] / total_units
            p_cooccur = cooccur_count / total_units
            
            # Avoid log(0)
            if p_tid1 > 0 and p_tid2 > 0 and p_cooccur > 0:
                pmi = np.log(p_cooccur / (p_tid1 * p_tid2))
                # Normalize PMI by -log(p_cooccur) to get NPMI in [-1, 1]
                npmi = pmi / (-np.log(p_cooccur))
                
                # Store only positive NPMI (positive correlation)
                if npmi > 0:
                    coactivation_matrix[(tid1, tid2)] = npmi
                    if tid1 != tid2:  # Make symmetric
                        coactivation_matrix[(tid2, tid1)] = npmi
        
        # Get unique token IDs
        token_ids = sorted(set(token_freq.keys()))
        
        return coactivation_matrix, token_ids, token_freq

    def _filter_frequent_tokens(
        self,
        coactivation_matrix: dict[tuple[int, int], float],
        token_ids: list[int],
        token_freq: dict[int, int],
    ) -> tuple[np.ndarray, list[int]]:
        """Filter co-activation matrix to top-F frequent tokens.
        
        Args:
            coactivation_matrix: Full co-activation matrix
            token_ids: All token IDs in matrix
            token_freq: Token frequency counts
            
        Returns:
            Tuple of (filtered_matrix, filtered_token_ids) where filtered_matrix
            is a dense numpy array of shape (F, F)
        """
        # Sort tokens by frequency and take top-F
        sorted_tokens = sorted(
            token_ids, key=lambda tid: token_freq.get(tid, 0), reverse=True
        )
        top_tokens = sorted_tokens[: self.top_tokens]
        
        # Build index mapping
        token_to_idx = {tid: idx for idx, tid in enumerate(top_tokens)}
        
        # Build filtered matrix as dense array
        n = len(top_tokens)
        filtered_matrix = np.zeros((n, n), dtype=np.float32)
        
        for (tid1, tid2), npmi in coactivation_matrix.items():
            if tid1 in token_to_idx and tid2 in token_to_idx:
                i = token_to_idx[tid1]
                j = token_to_idx[tid2]
                filtered_matrix[i, j] = npmi
        
        return filtered_matrix, top_tokens

    def _spectral_cluster(self, adjacency_matrix: np.ndarray) -> np.ndarray:
        """Apply spectral clustering to adjacency matrix.
        
        Computes normalized graph Laplacian, extracts bottom-K eigenvectors,
        and runs k-means clustering.
        
        Args:
            adjacency_matrix: Symmetric adjacency matrix (F x F)
            
        Returns:
            Cluster labels array of shape (F,)
        """
        # Handle empty or small matrices
        if adjacency_matrix.shape[0] == 0:
            return np.array([], dtype=np.int32)
        
        if adjacency_matrix.shape[0] < self.n_clusters:
            # Not enough tokens for full clustering - assign each to own cluster
            return np.arange(adjacency_matrix.shape[0], dtype=np.int32)
        
        # Compute degree matrix
        degrees = adjacency_matrix.sum(axis=1)
        
        # Avoid division by zero
        degrees = np.where(degrees > 0, degrees, 1)
        
        # Compute D^{-1/2}
        d_inv_sqrt = np.diag(1.0 / np.sqrt(degrees))
        
        # Compute normalized Laplacian: L = I - D^{-1/2} A D^{-1/2}
        identity = np.eye(adjacency_matrix.shape[0])
        normalized_adj = d_inv_sqrt @ adjacency_matrix @ d_inv_sqrt
        laplacian = identity - normalized_adj
        
        # Convert to sparse for efficient eigendecomposition
        laplacian_sparse = csr_matrix(laplacian)
        
        # Extract bottom-K eigenvectors (smallest eigenvalues)
        # For normalized Laplacian, smallest eigenvalues correspond to clusters
        try:
            k = min(self.n_clusters, adjacency_matrix.shape[0] - 1)
            eigenvalues, eigenvectors = eigsh(
                laplacian_sparse, k=k, which="SM", return_eigenvectors=True
            )
            
            # Run k-means on eigenvectors
            centroids, labels = kmeans2(eigenvectors, k, minit="points")
            
            return labels
            
        except Exception as e:
            # Fallback to simple frequency-based clustering if spectral fails
            warnings.warn(f"Spectral clustering failed: {e}. Using fallback.")
            # Assign tokens to clusters round-robin
            return np.arange(adjacency_matrix.shape[0]) % self.n_clusters

    def _infer_anchor_type(
        self, cluster_tokens: list[int], mode: str
    ) -> str:
        """Infer anchor type from cluster token IDs.
        
        Args:
            cluster_tokens: List of token IDs in cluster
            mode: 'code' or 'text'
            
        Returns:
            Anchor type: 'actor', 'relation', or 'feature'
        """
        # For code mode, most clusters are actors (classes, modules, functions)
        # Relations are handled separately via CODE_RELATION_ANCHORS
        if mode == "code":
            # Simple heuristic: most code tokens are actors
            return "actor"
        
        # For text mode, use more sophisticated heuristics
        # This is a simplified version - real implementation would use POS tagging
        # For now, default to feature
        return "feature"

    def _build_anchors_from_clusters(
        self,
        cluster_labels: np.ndarray,
        token_ids: list[int],
        token_freq: dict[int, int],
        mode: str,
    ) -> dict[str, Anchor]:
        """Build anchors from cluster assignments.
        
        Args:
            cluster_labels: Cluster label for each token
            token_ids: Token IDs corresponding to cluster labels
            token_freq: Token frequency counts
            mode: 'code' or 'text'
            
        Returns:
            Dict mapping anchor key to Anchor
        """
        anchors: dict[str, Anchor] = {}
        
        # Group tokens by cluster
        clusters: dict[int, list[int]] = defaultdict(list)
        for token_id, label in zip(token_ids, cluster_labels):
            clusters[label].append(token_id)
        
        # Build anchor for each cluster
        for cluster_id, cluster_token_ids in clusters.items():
            if not cluster_token_ids:
                continue
            
            # Sort tokens by frequency and take top-3 for labeling
            sorted_tokens = sorted(
                cluster_token_ids,
                key=lambda tid: token_freq.get(tid, 0),
                reverse=True,
            )
            top_tokens = sorted_tokens[:3]
            
            # Decode token IDs to strings
            # For SPLADE, we need to use the tokenizer vocab
            token_strs = []
            vocab = self.encoder.tokenizer.get_vocab()
            id_to_token = {v: k for k, v in vocab.items()}
            
            for tid in top_tokens:
                if tid in id_to_token:
                    token_str = id_to_token[tid]
                    # Clean up BERT subword markers
                    token_str = token_str.replace("##", "")
                    if token_str and not token_str.startswith("["):
                        token_strs.append(token_str.lower())
            
            if not token_strs:
                continue
            
            # Create anchor key from top token(s)
            key = "_".join(token_strs[:2]) if len(token_strs) > 1 else token_strs[0]
            key = f"cluster_{cluster_id}_{key}"[:50]  # Limit length
            
            # Skip if key would conflict with built-in anchors
            if mode == "code" and key in CODE_RELATION_ANCHORS:
                continue
            
            # Infer anchor type
            anchor_type = self._infer_anchor_type(cluster_token_ids, mode)
            
            # Create anchor
            anchors[key] = Anchor(
                key=key,
                type=anchor_type,
                tokens=token_strs[:5],  # Use top-5 tokens
                description=f"Auto-discovered {anchor_type} cluster {cluster_id}",
                parent=None,
            )
        
        return anchors
