const examples = [
  {
    title: "DR Auto Measurement",
    description: "This is a demo for DR Auto Measurement. You can upload a fundus image and the output will be the measurements of the optic disc and optic cup."
  },
  {
    title: "Generative Outpainting",
    description: "This is a demo for outpaint the image using Prod/Flux/Seedream. You can input an image and a ratio, and the model will generate a new image with the specified background expanded."
  },
  {
    title: "Expand Background V1",
    description: "This is a demo for expanding the background of images using Recraft. You can input an image and a ratio, and the model will generate a new image with the specified background expanded."
  },
  {
    title: "Expand Background V2",
    description: "This is a demo for expanding the background of images using FLUX. You can input an image and a ratio, and the model will generate a new image with the specified background expanded."
  },
  {
    title: "ROI Detection",
    description: "This is a demo for detecting regions of interest (ROI) in images using the BASNet model. You can input an image and a mask threshold, and the model will generate an output image with detected ROIs highlighted."
  },
  {
    title: "ROI Detection V2",
    description: "This is a demo for detecting regions of interest (ROI) in images using the DETR model. You can input an image and a score threshold, and the model will generate an output image with detected ROIs highlighted."
  },
  {
    title: "Background Synthesis",
    description: "This is a demo for background synthesis using Gemini & GPT. You can input an image of a product, and the model will generate a new image with an improved background suitable for e-commerce."
  }
];

const Examples = () => {
  return (
    <div className="max-w-6xl space-y-6 pb-8">
      <div>
        <h2 className="text-3xl font-bold text-zinc-900 dark:text-zinc-100 tracking-tight">Examples</h2>
        <p className="text-zinc-500 dark:text-zinc-400 mt-2">Explore demos powered by our customized models.</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-6">
        {examples.map((example, index) => (
          <div 
            key={index} 
            className="group bg-white dark:bg-zinc-900/60 border border-zinc-200 dark:border-zinc-800/60 rounded-2xl p-6 shadow-sm hover:shadow-md hover:border-zinc-300 dark:hover:border-zinc-700 transition-all cursor-pointer flex flex-col h-full"
          >
            <h3 className="text-lg font-semibold text-zinc-900 dark:text-zinc-100 mb-3 group-hover:text-orange-500 dark:group-hover:text-orange-400 transition-colors">
              {example.title}
            </h3>
            <p className="text-sm text-zinc-600 dark:text-zinc-400 leading-relaxed flex-1">
              {example.description}
            </p>
            <div className="mt-6 pt-4 border-t border-zinc-100 dark:border-zinc-800/80 flex items-center justify-between">
              <span className="text-xs font-medium text-orange-500 dark:text-orange-400">View Demo</span>
              <span className="text-zinc-400 group-hover:translate-x-1 transition-transform">→</span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default Examples;