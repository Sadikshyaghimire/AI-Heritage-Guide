import { useRef, useState } from "react";
import "./App.css";

const BASE_URL = "http://127.0.0.1:8000";

const API_URL = `${BASE_URL}/predict`;
const CHATBOT_URL = `${BASE_URL}/chatbot`;
const SIGNUP_URL = `${BASE_URL}/signup`;
const LOGIN_URL = `${BASE_URL}/login`;
const HERITAGE_INFO_URL = `${BASE_URL}/heritage-info`;

function App() {
  const fileInputRef = useRef(null);

  const [screen, setScreen] = useState("welcome");
  const [user, setUser] = useState(null);

  const [authMode, setAuthMode] = useState("login");
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");

  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const [question, setQuestion] = useState("");
  const [chatAnswer, setChatAnswer] = useState("");
  const [chatLoading, setChatLoading] = useState(false);

  const [authMessage, setAuthMessage] = useState("");
  const [scanHistory, setScanHistory] = useState([]);
  const [chatHistory, setChatHistory] = useState([]);
  const [historyLoading, setHistoryLoading] = useState(false);

  const [heritageList, setHeritageList] = useState({});
  const [selectedHeritage, setSelectedHeritage] = useState(null);

  const formatName = (value) =>
    value
      ? value.replaceAll("_", " ").replace(/\b\w/g, (c) => c.toUpperCase())
      : "";

  const handleAuth = async () => {
    setAuthMessage("");

    const url = authMode === "signup" ? SIGNUP_URL : LOGIN_URL;
    const body =
      authMode === "signup"
        ? { name, email, password }
        : { email, password };

    try {
      const response = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });

      const data = await response.json();

      if (!response.ok) {
        setAuthMessage(data.detail || "Authentication failed");
        return;
      }

      setUser(data.user);
      setScreen("dashboard");
    } catch {
      setAuthMessage("Could not connect to backend.");
    }
  };

  const handleImageChange = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    setPreview(URL.createObjectURL(file));
    setResult(null);
    setChatAnswer("");
    setQuestion("");
    setLoading(true);

    const formData = new FormData();
    formData.append("file", file);
    formData.append("user_email", user?.email || "");

    try {
      const response = await fetch(API_URL, {
        method: "POST",
        body: formData,
      });

      const data = await response.json();
      setResult(data);
      setScreen("scan");
    } catch {
      setResult({
        error: "Prediction failed. Make sure backend is running.",
      });
      setScreen("scan");
    } finally {
      setLoading(false);
    }
  };

  const handleAskQuestion = async () => {
    if (!question.trim() || !result?.predicted_landmark) return;

    setChatLoading(true);
    setChatAnswer("");

    try {
      const response = await fetch(CHATBOT_URL, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          landmark: result.predicted_landmark,
          question,
          user_email: user?.email || "",
        }),
      });

      const data = await response.json();
      setChatAnswer(data.answer);
    } catch {
      setChatAnswer("Chatbot failed. Make sure backend is running.");
    } finally {
      setChatLoading(false);
    }
  };

  const loadScanHistory = async () => {
    if (!user?.email) return;

    setHistoryLoading(true);
    setScreen("scanHistory");

    try {
      const response = await fetch(`${BASE_URL}/scan-history/${user.email}`);
      const data = await response.json();
      setScanHistory(data.history || []);
    } catch {
      setScanHistory([]);
    } finally {
      setHistoryLoading(false);
    }
  };

  const loadChatHistory = async () => {
    if (!user?.email) return;

    setHistoryLoading(true);
    setScreen("chatHistory");

    try {
      const response = await fetch(`${BASE_URL}/chat-history/${user.email}`);
      const data = await response.json();
      setChatHistory(data.history || []);
    } catch {
      setChatHistory([]);
    } finally {
      setHistoryLoading(false);
    }
  };

  const loadHeritageExplorer = async () => {
    try {
      const response = await fetch(HERITAGE_INFO_URL);
      const data = await response.json();
      setHeritageList(data);
      setSelectedHeritage(null);
      setScreen("heritageExplorer");
    } catch {
      alert("Could not load heritage information.");
    }
  };

  const resetScan = () => {
    setPreview(null);
    setResult(null);
    setQuestion("");
    setChatAnswer("");
    setScreen("dashboard");
  };

  const logout = () => {
    setUser(null);
    setScreen("welcome");
    setPreview(null);
    setResult(null);
    setQuestion("");
    setChatAnswer("");
    setScanHistory([]);
    setChatHistory([]);
    setHeritageList({});
    setSelectedHeritage(null);
  };

  if (screen === "welcome") {
    return (
      <div className="app welcomePage">
        <div className="heroCard">
          <img src="/icon-192.png" className="logo" alt="App logo" />
          <h1>AI Heritage Guide</h1>
          <p>
            Discover Kathmandu Valley heritage sites using AI-powered landmark
            recognition, cultural information, recommendations, and chatbot
            guidance.
          </p>

          <button onClick={() => setScreen("auth")} className="primaryButton">
            Get Started
          </button>
        </div>
      </div>
    );
  }

  if (screen === "auth") {
    return (
      <div className="app authPage">
        <div className="card">
          <img src="/icon-192.png" className="logo smallLogo" alt="App logo" />

          <h1>{authMode === "login" ? "Login" : "Create Account"}</h1>

          {authMode === "signup" && (
            <input
              className="input"
              placeholder="Full Name"
              value={name}
              onChange={(e) => setName(e.target.value)}
            />
          )}

          <input
            className="input"
            placeholder="Email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
          />

          <input
            className="input"
            placeholder="Password"
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
          />

          {authMessage && <p className="error">{authMessage}</p>}

          <button className="primaryButton" onClick={handleAuth}>
            {authMode === "login" ? "Login" : "Sign Up"}
          </button>

          <button
            className="linkButton"
            onClick={() =>
              setAuthMode(authMode === "login" ? "signup" : "login")
            }
          >
            {authMode === "login"
              ? "New user? Create an account"
              : "Already have an account? Login"}
          </button>
        </div>
      </div>
    );
  }

  if (screen === "dashboard") {
    return (
      <div className="app">
        <header className="header">
          <img src="/icon-192.png" className="logo smallLogo" alt="App logo" />
          <h1>AI Heritage Guide</h1>
          <p>Welcome, {user?.name}</p>
        </header>

        <main className="card">
          <h2>Dashboard</h2>
          <p>
            Scan heritage landmarks, explore cultural sites, view your activity
            history, and ask the AI guide for cultural information.
          </p>

          <button
            className="primaryButton"
            onClick={() => fileInputRef.current.click()}
          >
            Scan / Upload Heritage Image
          </button>

          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            onChange={handleImageChange}
            hidden
          />

          <button className="secondaryButton" onClick={loadHeritageExplorer}>
            Heritage Explorer
          </button>

          <button className="secondaryButton" onClick={loadScanHistory}>
            View Scan History
          </button>

          <button className="secondaryButton" onClick={loadChatHistory}>
            View Chat History
          </button>

          <button className="secondaryButton" onClick={logout}>
            Logout
          </button>
        </main>
      </div>
    );
  }

  if (screen === "heritageExplorer") {
    return (
      <div className="app">
        <header className="header">
          <h1>Heritage Explorer</h1>
          <p>Browse cultural and historical sites supported by the app.</p>
        </header>

        <main className="card">
          {!selectedHeritage && (
            <>
              {Object.entries(heritageList).map(([key, site]) => (
               <div
  className="explorerItem"
  key={key}
  onClick={() => setSelectedHeritage({ key, ...site })}
>
  <h3>{site.name}</h3>
  <p>{site.location}</p>
  <span>{site.type}</span>

  <div className={site.recognition_enabled ? "recognitionYes" : "recognitionNo"}>
    {site.recognition_enabled ? "AI Recognition Available" : "Information Only"}
  </div>
</div>
              ))}
            </>
          )}

          {selectedHeritage && (
            <div className="resultBox">
              <h2>{selectedHeritage.name}</h2>

              <p>
                <strong>Location:</strong> {selectedHeritage.location}
              </p>

              <p>
  <strong>Recognition:</strong>{" "}
  {selectedHeritage.recognition_enabled
    ? "AI recognition available"
    : "Information only"}
</p>

              <p>
                <strong>Description:</strong>
              </p>

              <p>{selectedHeritage.description}</p>

              <hr />

              <h3>Recommended Places</h3>
              {selectedHeritage.recommended?.map((place, index) => (
                <div key={index} className="prediction">
                  {formatName(place)}
                </div>
              ))}

              <button
                className="secondaryButton"
                onClick={() => setSelectedHeritage(null)}
              >
                Back to Heritage List
              </button>
            </div>
          )}

          <button
            className="secondaryButton"
            onClick={() => {
              setSelectedHeritage(null);
              setScreen("dashboard");
            }}
          >
            Back to Dashboard
          </button>
        </main>
      </div>
    );
  }

  if (screen === "scanHistory") {
    return (
      <div className="app">
        <header className="header">
          <h1>Scan History</h1>
          <p>{user?.email}</p>
        </header>

        <main className="card">
          {historyLoading && <p className="loading">Loading history...</p>}

          {!historyLoading && scanHistory.length === 0 && (
            <p>No scan history found yet.</p>
          )}

          {scanHistory.map((item, index) => (
            <div className="resultBox" key={index}>
              <h3>{formatName(item.predicted_landmark)}</h3>
              <p>
                <strong>Confidence:</strong>{" "}
                {(item.confidence * 100).toFixed(2)}%
              </p>
              <p>
                <strong>Date:</strong> {item.created_at}
              </p>
            </div>
          ))}

          <button
            className="secondaryButton"
            onClick={() => setScreen("dashboard")}
          >
            Back to Dashboard
          </button>
        </main>
      </div>
    );
  }

  if (screen === "chatHistory") {
    return (
      <div className="app">
        <header className="header">
          <h1>Chat History</h1>
          <p>{user?.email}</p>
        </header>

        <main className="card">
          {historyLoading && <p className="loading">Loading history...</p>}

          {!historyLoading && chatHistory.length === 0 && (
            <p>No chat history found yet.</p>
          )}

          {chatHistory.map((item, index) => (
            <div className="resultBox" key={index}>
              <h3>{formatName(item.landmark)}</h3>
              <p>
                <strong>Question:</strong> {item.question}
              </p>
              <p>
                <strong>Answer:</strong> {item.answer}
              </p>
              <p>
                <strong>Date:</strong> {item.created_at}
              </p>
            </div>
          ))}

          <button
            className="secondaryButton"
            onClick={() => setScreen("dashboard")}
          >
            Back to Dashboard
          </button>
        </main>
      </div>
    );
  }

  return (
    <div className="app">
      <header className="header">
        <img src="/icon-192.png" className="logo smallLogo" alt="App logo" />
        <h1>AI Heritage Guide</h1>
        <p>Welcome, {user?.name}</p>
      </header>

      <main className="card">
        <button
          className="cameraButton"
          onClick={() => fileInputRef.current.click()}
        >
          Scan Another Image
        </button>

        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          onChange={handleImageChange}
          hidden
        />

        {preview && (
          <div className="previewBox">
            <img src={preview} alt="Selected heritage" />
          </div>
        )}

        {loading && <p className="loading">Recognizing landmark...</p>}

        {result && !result.error && (
          <div className="resultBox">

            <h2>{result.info?.name || formatName(result.predicted_landmark)}</h2>

            <p>
              <strong>Confidence:</strong>{" "}
              {(result.confidence * 100).toFixed(2)}%
            </p>

            <p>
              <strong>Location:</strong> {result.info?.location}
            </p>

            <p>
              <strong>Type:</strong> {result.info?.type}
            </p>

            <p>
              <strong>Description:</strong>
            </p>

            <p>{result.info?.description}</p>

            <hr />

            <h3>Recommended Places</h3>
            {result.info?.recommended?.map((place, index) => (
              <div key={index} className="prediction">
                {formatName(place)}
              </div>
            ))}

            <hr />

            <h3>Top Predictions</h3>
            {result.top_3_predictions.map((item, index) => (
              <div className="prediction" key={index}>
                <span>{formatName(item.landmark)}</span>
                <span>{(item.confidence * 100).toFixed(2)}%</span>
              </div>
            ))}

            <hr />

            <h3>Ask About This Heritage Site</h3>

            <div className="chatBox">
              <input
                type="text"
                placeholder="Ask a question..."
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
              />

              <button onClick={handleAskQuestion}>Ask</button>
            </div>

            {chatLoading && <p className="loading">Thinking...</p>}

            {chatAnswer && (
              <div className="chatAnswer">
                <strong>Guide:</strong>
                <p>{chatAnswer}</p>
              </div>
            )}
          </div>
        )}

        {result?.error && <p className="error">{result.error}</p>}

        <button className="secondaryButton" onClick={resetScan}>
          Back to Dashboard
        </button>

        <button className="secondaryButton" onClick={logout}>
          Logout
        </button>
      </main>
    </div>
  );
}

export default App;