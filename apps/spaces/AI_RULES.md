# AI Rules & Guidelines

This document serves as a guideline for AI assistants and developers working on this project. It defines the tech stack and the strict rules regarding library usage and architectural decisions.

## Tech Stack

- **React**: Core UI library for building component-driven interfaces.
- **TypeScript**: Used strictly across the entire project for static typing and better developer experience.
- **Vite**: Ultra-fast build tool and development server.
- **React Router**: Standard library for client-side routing.
- **Tailwind CSS**: Utility-first CSS framework for all styling and responsive design.
- **shadcn/ui & Radix UI**: Accessible, customizable pre-built UI components.
- **Lucide React**: Primary icon library.
- **ESLint**: Code linting and formatting enforcement.

## Library & Development Rules

### 1. Styling
- **Use Tailwind CSS exclusively:** All component styling, layouts, and spacing must be handled via Tailwind CSS utility classes. Avoid writing custom CSS in `.css` files unless absolutely necessary for global resets or specific animations not supported by Tailwind.
- **Responsive Design:** Always utilize Tailwind's responsive prefixes (`sm:`, `md:`, `lg:`, etc.) to ensure a mobile-first responsive design.

### 2. UI Components
- **Use shadcn/ui:** Before building a custom component (like Buttons, Modals, Dropdowns), check if a shadcn/ui component exists and use it. 
- **Icons:** Always use `lucide-react` for icons. Do not import SVG files directly unless they are custom brand assets.

### 3. Routing
- **React Router:** All application routing must be handled by React Router.
- **Route Definition:** Maintain all route definitions centrally in `src/App.tsx`.

### 4. File Structure & Architecture
- **Pages vs Components:** Place full views/pages in `src/pages/` and reusable UI pieces in `src/components/`.
- **Modularity:** Keep components small and focused (ideally under 100 lines). If a file grows too large, refactor it into smaller sub-components.
- **Default Page:** The main landing page should be located at `src/pages/Index.tsx`.

### 5. State & Data Handling
- **Local State:** Use React's built-in hooks (`useState`, `useReducer`, `useContext`) for local and UI state.
- **Side Effects:** Keep `useEffect` usage minimal and focused. 
- **Error Handling:** Do not aggressively swallow errors with `try/catch` blocks unless specific fallback UI is requested. Let errors bubble up during development to catch underlying issues.