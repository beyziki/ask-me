import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";
import { AuthProvider, useAuth } from "./context/AuthContext";
import { DocumentsProvider } from "./context/DocumentsContext";
import Layout from "./components/Layout";
import LoginPage from "./pages/LoginPage";
import UploadPage from "./pages/UploadPage";
import AskPage from "./pages/AskPage";
import QuizPage from "./pages/QuizPage";
import SummaryPage from "./pages/SummaryPage";
import CodePage from "./pages/CodePage";
import SplitViewPage from "./pages/SplitViewPage";

function LoginRoute() {
  const { user } = useAuth();
  if (user) return <Navigate to="/upload" replace />;
  return <LoginPage />;
}

function App() {
  return (
    <AuthProvider>
      <DocumentsProvider>
        <BrowserRouter>
          <Routes>
            <Route path="/login" element={<LoginRoute />} />

            <Route element={<Layout />}>
              <Route index element={<Navigate to="/upload" replace />} />
              <Route path="/upload" element={<UploadPage />} />
              <Route path="/ask" element={<AskPage />} />
              <Route path="/quiz" element={<QuizPage />} />
              <Route path="/summary" element={<SummaryPage />} />
              <Route path="/split" element={<SplitViewPage />} />
              <Route path="/code" element={<CodePage />} />
            </Route>

            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </BrowserRouter>
      </DocumentsProvider>
    </AuthProvider>
  );
}

export default App;
