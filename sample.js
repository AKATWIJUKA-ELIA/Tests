const query = async  (data)=> {
	const response = await fetch(
		"https://router.huggingface.co/hf-inference/models/sentence-transformers/all-MiniLM-L6-v2/pipeline/sentence-similarity",
		{
			headers: {
				Authorization:  "Bearer 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",
				"Content-Type": "application/json",
			},
			method: "POST",
			body: JSON.stringify(data),
		}
	);
	const result = await response.json();
	return result;
}      

query({ inputs: {
    "source_sentence": "skating shoes very comfortable.",
    "sentences": [
        "skating shoes",
        "comfortable open shoes ",
        "hiking footware",
          "Today is a sunny day"
    ]
} }).then((response) => {
    console.log(JSON.stringify(response));
});
