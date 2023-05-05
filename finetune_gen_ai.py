from transformers import GPT2Tokenizer, GPT2LMHeadModel, BloomForCausalLM,BloomTokenizerFast
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import numpy as np
import torch
import csv


#The all important prompt!
prompts = ["Who is Azahar?"]

#Generated Text Length
MAX_SENTENCE_LEN = 30

#Epochs
EPOCHS = 10

#Number of samples (in this case blog posts) to take
SAMPLES = 30

# Batch size and learning rate
BATCH_SIZE = 10
LEARNING_RATE = 1e-4

# Choose 0 for Bloom, 1 for GPT2
model_index = 1

# Do not change unless you are adding/removing model types
# ----------------------- START -----------------------------
model_class = [BloomForCausalLM, GPT2LMHeadModel][model_index]
tokenizer_class = [BloomTokenizerFast, GPT2Tokenizer][model_index]

# Model name and friendly name (for saving the fine-tuned model)
model_name = ['bigscience/bloom-560m',"gpt2"][model_index] #'bigscience/bloom-560m' #gpt2  
friendly_model_name =  ["bloom_560m","gpt2"][model_index]
# ----------------------------- END ------------------------

# Max chunk of text to take from the blog post fine tuning - larger means longer training
MAX_SIZE = 500

# Take chunks from beginning and end of blog posts
end_start = [MAX_SIZE,-MAX_SIZE]


#Change to where you want to save the finetuned model.
# Directories needed: models, results.
CUSTOM_MODEL_FILE_PATH = f"./models/model_gen_short_blog_{MAX_SIZE}_{friendly_model_name}"
FINE_TUNE_DATA_LOCATION = "data/posts/blog_post_index.txt"

STANDARD_MODEL_DIST_FILEPATH = f"results/standard_model_{MAX_SIZE}_{friendly_model_name}.txt"
CUSTOM_MODEL_DIST_FILEPATH = f"results/custom_model_{MAX_SIZE}_{friendly_model_name}.txt"

GEN_OUTPUT_CUST_MODEL_PATH = f"results/gen_custom_{MAX_SIZE}_{friendly_model_name}.txt"
GEN_OUTPUT_STANDARD_MODEL_PATH = f"results/gen_standard_{MAX_SIZE}_{friendly_model_name}.txt"





def build_distance(outputs,model_type):
    distances = []
   
    for idx,i in enumerate(outputs):
        for jdx,j in enumerate(outputs[:idx]):
            if idx != jdx:
                distances.append(np.sum(np.abs(i-j)))
    
    np.array(distances).tofile(f"distance_data_{model_type}.csv")

    return distances

def build_comp_distance(output1,output2,model_type):
    distances = []
   
    for idx,i in enumerate(output1):
        for jdx,j in enumerate(output2):
                distances.append(np.sum(np.abs(i-j)))
    
    np.array(distances).tofile(f"distance_data_{model_type}.csv")

    return distances

# Process an input given a model and a tokenizer
def proc(text,model,tokenizer):
	tokens = tokenizer(text,return_tensors='pt')["input_ids"]
	empty = np.zeros((MAX_SENTENCE_LEN,1))
	output = model.generate(tokens,max_length=MAX_SENTENCE_LEN, do_sample=True,pad_token_id=tokenizer.eos_token_id)

	for idx, j in enumerate(output[0]):
		empty[idx]=j
	return empty, output

# Embedding to text
def get_text(input,tokenizer):
	decoded = tokenizer.decode(input, skip_special_tokens=True)
	return decoded

# Pretty print
def print_text(text,model_name="<gen output>"):
	print(f"{model_name}:",text,"\n")


# Generate and extract output to file.
def generate(prompt,model,tokenizer,fh):
	out, txt = proc(prompt,model,tokenizer)
	
	#Uncomment to get text output
	try:
		text = get_text(txt[0],tokenizer)
		text = text.replace(prompt,"")
		fh.write(text+"\n")
	except:
		print("Error:",text)
	return out	


model = None
tokenizer = tokenizer_class.from_pretrained(model_name)
try:
	# Model exists - load it
	model = torch.load(CUSTOM_MODEL_FILE_PATH)
except:
	
	# Build new model
	
	#Load base model
	model = model_class.from_pretrained(model_name)

	#Load custom text file and prepare input tokens

	#The index text file containing all the training data text file names
	#One training data text file per blog post
	text = csv.reader(open(FINE_TUNE_DATA_LOCATION))
	files = []
	for l in next(text):
		if l:
			files.append(l)
	
	print("Files to process:",len(files))

	#Create dataset using filenames obtained from index file
	dataset = load_dataset("text", data_files={"text":files})
	
	dataset = dataset.map(lambda examples: tokenizer(examples["text"]), batched=True)

	#Train the model

	train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

	#Setup optimizer to fine tune the model.
	optimizer = torch.optim.AdamW(model.parameters(),lr=LEARNING_RATE)

	
	for epoch in range(EPOCHS):
		print("Epoch:",epoch)
		for idx,batch in enumerate(np.random.choice(dataset['text'],SAMPLES,replace=False)):
			size = len(batch["input_ids"])
			
			print(epoch," Docs done:",idx," Doc size (tokens):",size)
			optimizer.zero_grad()
			if size < MAX_SIZE:
				tens = torch.tensor(batch['input_ids'])
				tens = torch.reshape(tens,[size,1])
			else:
				# Choose whether to take sample from the start or the end of the text (blog post)
				val = np.random.choice(end_start,1)[0]
				if val > 0:
					tens = torch.tensor(batch['input_ids'][:val])
				else:
					tens = torch.tensor(batch['input_ids'][val:])
				
				tens = torch.reshape(tens,[MAX_SIZE,1])
			input_ids = tens
			labels = tens
		
			# The actual backpropagation training
			outputs = model(input_ids, labels = labels)
			loss = outputs[0]
			loss.backward()
			optimizer.step()

	#Save trained model
	torch.save(model,CUSTOM_MODEL_FILE_PATH)

#Load base model for comparison.
model2 = model_class.from_pretrained(model_name)

out_ft = []
out_standard = []

with open(GEN_OUTPUT_CUST_MODEL_PATH,'w') as gen_custom, open(GEN_OUTPUT_STANDARD_MODEL_PATH,'w') as gen_standard:
	for i in range(100):
		print(i)
		for p in prompts:
			
			out_ft.append(generate(p,model,tokenizer,gen_custom))
			out_standard.append(generate(p,model2,tokenizer,gen_standard))
			

dist_ft = build_distance(out_ft,"fine_tuned")
dist_standard = build_distance(out_standard,"standard")


np.array(dist_standard).tofile(STANDARD_MODEL_DIST_FILEPATH)
np.array(dist_ft).tofile(CUSTOM_MODEL_DIST_FILEPATH)

# Plot out generated output distance hist
from matplotlib import pyplot as plt
fig,ax = plt.subplots(2,1,sharex=True)
ax[0].hist(dist_ft,bins=40)
ax[0].set_title("Fine-tuned Model")
ax[1].hist(dist_standard,bins=40)
ax[1].set_title("Standard Model")

plt.show()

