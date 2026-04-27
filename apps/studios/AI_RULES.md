# AI Rules and Tech Stack

## Tech Stack
* **React:** Core UI library (Version 19+).
* **TypeScript:** For static typing, interfaces, and overall type safety.
* **Vite:** Build tool and development server.
* **Tailwind CSS:** Utility-first CSS framework for all styling and layout needs.
* **shadcn/ui & Radix UI:** For highly accessible, prebuilt UI components.
* **React Router:** For client-side routing and navigation.
* **Lucide React:** Standard library for SVG icons.

## Library Rules and Guidelines

### 1. Styling
* **Rule:** ONLY use Tailwind CSS for styling. 
* **Details:** Avoid writing custom CSS in `.css` files unless absolutely necessary for global resets or specific animations not covered by Tailwind. Rely on Tailwind utility classes.

### 2. UI Components
* **Rule:** Always prioritize `shadcn/ui` components.
* **Details:** Whenever a standard UI element is needed (Buttons, Dialogs, Dropdowns, etc.), use the prebuilt `shadcn/ui` component. Do not build custom versions of these from scratch unless the design specifically demands it. Note that `shadcn/ui` components should not be heavily edited; create wrappers if needed.

### 3. Routing
* **Rule:** Use React Router (`react-router-dom`) for all page navigation.
* **Details:** Keep all route definitions centralized within `src/App.tsx`. Place page components inside the `src/pages/` directory.

### 4. Icons
* **Rule:** Use `lucide-react`.
* **Details:** Do not import SVGs manually or use other icon libraries. Import the required icon directly from the `lucide-react` package.

### 5. File Structure
* **Rule:** Strictly separate pages and components.
* **Details:** 
  * Reusable UI elements go in `src/components/`.
  * Route-level views go in `src/pages/`.
  * The default landing page must be `src/pages/Index.tsx`.
  * All directory names must be lowercase. File names should generally be PascalCase for components.

### 6. Code Simplicity
* **Rule:** Do not overengineer.
* **Details:** Write simple, elegant code. Do not implement complex error handling, fallback mechanisms, or unnecessary abstractions unless explicitly requested. Keep components under 100 lines when possible.