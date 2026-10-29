/* ================================================================
   LoanTrack — Frontend Logic
   Login + Prediction + Bank Recommendation
   All currency in ₹ INR
   ================================================================ */

(function () {
    "use strict";

    // ═══════════════════════════════════════════════════════
    // DOM References
    // ═══════════════════════════════════════════════════════

    // Auth
    const loginCard     = document.getElementById("login-card");
    const loginForm     = document.getElementById("login-form");
    const signupForm    = document.getElementById("signup-form");
    const tabLogin      = document.getElementById("tab-login");
    const tabSignup     = document.getElementById("tab-signup");
    const tabIndicator  = document.getElementById("tab-indicator");
    const loginError    = document.getElementById("login-error");
    const signupError   = document.getElementById("signup-error");
    const btnLogin      = document.getElementById("btn-login");
    const btnSignup     = document.getElementById("btn-signup");
    const headerUser    = document.getElementById("header-user");
    const headerAvatar  = document.getElementById("header-avatar");
    const headerUsername = document.getElementById("header-username");
    const btnLogout     = document.getElementById("btn-logout");

    // Loan form
    const formCard    = document.getElementById("form-card");
    const form        = document.getElementById("loan-form");
    const btnPredict  = document.getElementById("btn-predict");

    // Results
    const resultCard       = document.getElementById("result-card");
    const resultBadge      = document.getElementById("result-badge");
    const resultIcon       = document.getElementById("result-icon");
    const resultStatus     = document.getElementById("result-status");
    const gaugeValue       = document.getElementById("gauge-value");
    const gaugeFill        = document.getElementById("gauge-fill");
    const gaugeDot         = document.getElementById("gauge-dot");
    const detailStatus     = document.getElementById("detail-status");
    const detailProbability = document.getElementById("detail-probability");
    const detailLoanInr    = document.getElementById("detail-loan-inr");
    const btnBankOffers    = document.getElementById("btn-bank-offers");
    const btnReset         = document.getElementById("btn-reset");

    // Bank recommendation
    const bankCard       = document.getElementById("bank-card");
    const bankOffersList = document.getElementById("bank-offers-list");
    const eligStatus     = document.getElementById("elig-status");
    const eligScore      = document.getElementById("elig-score");
    const eligAmount     = document.getElementById("elig-amount");
    const eligTerm       = document.getElementById("elig-term");
    const btnBackResults = document.getElementById("btn-back-results");
    const btnNewFromBank = document.getElementById("btn-new-from-bank");

    // State
    let currentUser = null;
    let lastPrediction = null;
    let lastFormData = null;

    // ═══════════════════════════════════════════════════════
    // INR Formatting
    // ═══════════════════════════════════════════════════════
    function formatINR(amount) {
        if (amount >= 10000000) {
            return "₹" + (amount / 10000000).toFixed(2) + " Cr";
        } else if (amount >= 100000) {
            return "₹" + (amount / 100000).toFixed(2) + " L";
        } else {
            return "₹" + amount.toLocaleString("en-IN");
        }
    }

    function formatINRFull(amount) {
        return "₹" + amount.toLocaleString("en-IN");
    }

    // ═══════════════════════════════════════════════════════
    // AUTH TABS
    // ═══════════════════════════════════════════════════════
    tabLogin.addEventListener("click", () => switchTab("login"));
    tabSignup.addEventListener("click", () => switchTab("signup"));

    function switchTab(tab) {
        if (tab === "login") {
            tabLogin.classList.add("active");
            tabSignup.classList.remove("active");
            tabIndicator.classList.remove("right");
            loginForm.classList.remove("hidden");
            signupForm.classList.add("hidden");
        } else {
            tabSignup.classList.add("active");
            tabLogin.classList.remove("active");
            tabIndicator.classList.add("right");
            signupForm.classList.remove("hidden");
            loginForm.classList.add("hidden");
        }
        loginError.textContent = "";
        signupError.textContent = "";
    }

    // ═══════════════════════════════════════════════════════
    // PASSWORD VISIBILITY TOGGLE
    // ═══════════════════════════════════════════════════════
    document.getElementById("toggle-login-pw").addEventListener("click", () => {
        const input = document.getElementById("login-password");
        input.type = input.type === "password" ? "text" : "password";
    });
    document.getElementById("toggle-signup-pw").addEventListener("click", () => {
        const input = document.getElementById("signup-password");
        input.type = input.type === "password" ? "text" : "password";
    });

    // ═══════════════════════════════════════════════════════
    // PASSWORD STRENGTH METER
    // ═══════════════════════════════════════════════════════
    const signupPassword = document.getElementById("signup-password");
    const strengthFill   = document.getElementById("strength-fill");
    const strengthText   = document.getElementById("strength-text");

    signupPassword.addEventListener("input", () => {
        const val = signupPassword.value;
        let score = 0;
        if (val.length >= 6) score++;
        if (val.length >= 10) score++;
        if (/[A-Z]/.test(val)) score++;
        if (/[0-9]/.test(val)) score++;
        if (/[^A-Za-z0-9]/.test(val)) score++;

        const levels = [
            { width: "0%",   color: "transparent",  text: "" },
            { width: "20%",  color: "#ef4444",      text: "Weak" },
            { width: "40%",  color: "#f59e0b",      text: "Fair" },
            { width: "60%",  color: "#f59e0b",      text: "Good" },
            { width: "80%",  color: "#22c55e",      text: "Strong" },
            { width: "100%", color: "#22c55e",      text: "Excellent" },
        ];
        const level = levels[score];
        strengthFill.style.width = level.width;
        strengthFill.style.background = level.color;
        strengthText.textContent = level.text;
        strengthText.style.color = level.color;
    });

    // ═══════════════════════════════════════════════════════
    // LOCAL STORAGE AUTH (client-side demo)
    // ═══════════════════════════════════════════════════════
    function getUsers() {
        try {
            return JSON.parse(
                localStorage.getItem("loantrack_users") ||
                localStorage.getItem("loanpredict_users") ||
                "{}"
            );
        } catch { return {}; }
    }
    function saveUsers(users) {
        localStorage.setItem("loantrack_users", JSON.stringify(users));
    }

    // ── SIGNUP ──
    signupForm.addEventListener("submit", (e) => {
        e.preventDefault();
        signupError.textContent = "";

        const name     = document.getElementById("signup-name").value.trim();
        const email    = document.getElementById("signup-email").value.trim().toLowerCase();
        const password = document.getElementById("signup-password").value;
        const confirm  = document.getElementById("signup-confirm").value;

        if (password !== confirm) {
            signupError.textContent = "Passwords do not match";
            return;
        }
        if (password.length < 6) {
            signupError.textContent = "Password must be at least 6 characters";
            return;
        }

        const users = getUsers();
        if (users[email]) {
            signupError.textContent = "An account with this email already exists";
            return;
        }

        users[email] = { name, email, password };
        saveUsers(users);

        // Auto-login after signup
        loginUser({ name, email });
    });

    // ── LOGIN ──
    loginForm.addEventListener("submit", (e) => {
        e.preventDefault();
        loginError.textContent = "";

        const email    = document.getElementById("login-email").value.trim().toLowerCase();
        const password = document.getElementById("login-password").value;

        const users = getUsers();

        if (!users[email]) {
            loginError.textContent = "No account found with this email";
            return;
        }
        if (users[email].password !== password) {
            loginError.textContent = "Incorrect password";
            return;
        }

        loginUser(users[email]);
    });

    // ── Social login (demo) ──
    document.getElementById("btn-google").addEventListener("click", () => {
        loginUser({ name: "Google User", email: "user@gmail.com" });
    });
    document.getElementById("btn-github").addEventListener("click", () => {
        loginUser({ name: "GitHub User", email: "user@github.com" });
    });

    function loginUser(user) {
        currentUser = user;
        localStorage.setItem("loantrack_session", JSON.stringify(user));

        // Update header
        headerAvatar.textContent = user.name.charAt(0).toUpperCase();
        headerUsername.textContent = user.name;
        headerUser.classList.remove("hidden");

        // Switch to form
        loginCard.classList.add("hidden");
        formCard.classList.remove("hidden");
    }

    // ── LOGOUT ──
    btnLogout.addEventListener("click", () => {
        currentUser = null;
        localStorage.removeItem("loantrack_session");
        localStorage.removeItem("loanpredict_session");
        headerUser.classList.add("hidden");
        formCard.classList.add("hidden");
        resultCard.classList.add("hidden");
        bankCard.classList.add("hidden");
        loginCard.classList.remove("hidden");
        loginForm.reset();
        signupForm.reset();
        switchTab("login");
    });

    // ── Auto-login on page load ──
    (function checkSession() {
        try {
            const session = JSON.parse(
                localStorage.getItem("loantrack_session") ||
                localStorage.getItem("loanpredict_session")
            );
            if (session && session.name && session.email) {
                loginUser(session);
            }
        } catch {}
    })();

    // ═══════════════════════════════════════════════════════
    // LOAN FORM SUBMISSION
    // ═══════════════════════════════════════════════════════
    form.addEventListener("submit", async (e) => {
        e.preventDefault();
        btnPredict.classList.add("loading");
        btnPredict.disabled = true;

        const data = Object.fromEntries(new FormData(form));

        // Cast numeric fields
        data.ApplicantIncome   = parseFloat(data.ApplicantIncome);
        data.CoapplicantIncome = parseFloat(data.CoapplicantIncome);
        data.LoanAmount        = parseFloat(data.LoanAmount);
        data.Loan_Amount_Term  = parseFloat(data.Loan_Amount_Term);
        data.Credit_History    = parseFloat(data.Credit_History);

        lastFormData = { ...data };

        try {
            const res = await fetch("/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(data),
            });

            if (!res.ok) {
                const err = await res.json();
                throw new Error(err.detail || "Prediction failed");
            }

            const result = await res.json();
            lastPrediction = result;
            showResult(result);
        } catch (err) {
            alert("⚠️  " + err.message);
        } finally {
            btnPredict.classList.remove("loading");
            btnPredict.disabled = false;
        }
    });

    // ═══════════════════════════════════════════════════════
    // DISPLAY RESULT
    // ═══════════════════════════════════════════════════════
    function showResult(result) {
        const approved = result.status === "APPROVED";
        const prob     = result.probability;

        // Badge
        resultBadge.className = "result-badge " + (approved ? "approved" : "rejected");
        resultIcon.innerHTML  = approved
            ? '<path d="M20 6L9 17l-5-5"/>'
            : '<path d="M18 6L6 18"/><path d="M6 6l12 12"/>';
        resultStatus.textContent = approved ? "APPROVED" : "REJECTED";

        // Detail pills
        detailStatus.textContent      = approved ? "Eligible ✓" : "Not Eligible ✗";
        detailProbability.textContent = result.probability_pct;

        // Loan amount in INR
        if (lastFormData) {
            const loanAmountINR = lastFormData.LoanAmount * 1000;
            detailLoanInr.textContent = formatINR(loanAmountINR);
        }

        // Gauge animation
        animateGauge(prob);

        // Swap cards
        formCard.classList.add("hidden");
        resultCard.classList.remove("hidden");
    }

    // ═══════════════════════════════════════════════════════
    // GAUGE SVG ANIMATION
    // ═══════════════════════════════════════════════════════
    function animateGauge(prob) {
        const ARC_LENGTH = 251.3;
        const offset = ARC_LENGTH * (1 - prob);

        gaugeFill.style.strokeDashoffset = ARC_LENGTH;
        requestAnimationFrame(() => {
            gaugeFill.style.strokeDashoffset = offset;
        });

        const angle = Math.PI * (1 - prob);
        const cx = 100 + 80 * Math.cos(angle);
        const cy = 100 - 80 * Math.sin(angle);
        gaugeDot.setAttribute("cx", cx);
        gaugeDot.setAttribute("cy", cy);

        animateCounter(gaugeValue, 0, prob * 100, 1000);
    }

    function animateCounter(el, from, to, duration) {
        const start = performance.now();
        function step(now) {
            const t = Math.min((now - start) / duration, 1);
            const eased = 1 - Math.pow(1 - t, 3);
            const val = from + (to - from) * eased;
            el.textContent = val.toFixed(1) + "%";
            if (t < 1) requestAnimationFrame(step);
        }
        requestAnimationFrame(step);
    }

    // ═══════════════════════════════════════════════════════
    // BANK RECOMMENDATION ENGINE
    // ═══════════════════════════════════════════════════════

    // Indian bank data with current typical rates (2026)
    const INDIAN_BANKS = [
        {
            name: "State Bank of India (SBI)",
            type: "PSU",
            minRate: 8.50, maxRate: 10.15,
            minIncome: 25000, maxLoan: 10000000,
            features: ["Lowest base rate", "Max 30yr tenure", "Govt-backed"],
            processingFee: "0.35%",
            logo: "🏦"
        },
        {
            name: "HDFC Bank",
            type: "Private",
            minRate: 8.75, maxRate: 9.65,
            minIncome: 30000, maxLoan: 15000000,
            features: ["Fast approval", "Doorstep service", "Digital process"],
            processingFee: "0.50%",
            logo: "🏛️"
        },
        {
            name: "ICICI Bank",
            type: "Private",
            minRate: 8.75, maxRate: 9.85,
            minIncome: 25000, maxLoan: 12000000,
            features: ["Instant approval", "Flexible EMI", "Balance transfer"],
            processingFee: "0.50%",
            logo: "🏢"
        },
        {
            name: "Bank of Baroda",
            type: "PSU",
            minRate: 8.40, maxRate: 10.65,
            minIncome: 20000, maxLoan: 10000000,
            features: ["Lowest processing fee", "No hidden charges", "Rural reach"],
            processingFee: "0.25%",
            logo: "🏦"
        },
        {
            name: "Punjab National Bank (PNB)",
            type: "PSU",
            minRate: 8.45, maxRate: 10.25,
            minIncome: 20000, maxLoan: 8000000,
            features: ["Low income threshold", "Wide branch network", "NRI loans"],
            processingFee: "0.35%",
            logo: "🏦"
        },
        {
            name: "Axis Bank",
            type: "Private",
            minRate: 8.75, maxRate: 10.30,
            minIncome: 30000, maxLoan: 10000000,
            features: ["Quick disbursal", "Overdraft facility", "Top-up loan"],
            processingFee: "0.50%",
            logo: "🏛️"
        },
        {
            name: "Kotak Mahindra Bank",
            type: "Private",
            minRate: 8.85, maxRate: 9.80,
            minIncome: 35000, maxLoan: 10000000,
            features: ["Zero foreclosure", "Doorstep pickup", "Minimal docs"],
            processingFee: "0.50%",
            logo: "🏛️"
        },
        {
            name: "Union Bank of India",
            type: "PSU",
            minRate: 8.35, maxRate: 10.55,
            minIncome: 18000, maxLoan: 7500000,
            features: ["Very low rates", "Flexible tenure", "Easy docs"],
            processingFee: "0.30%",
            logo: "🏦"
        },
        {
            name: "LIC Housing Finance",
            type: "NBFC",
            minRate: 8.50, maxRate: 10.75,
            minIncome: 20000, maxLoan: 15000000,
            features: ["High max loan", "LIC benefit link", "30yr tenure"],
            processingFee: "0.50%",
            logo: "🛡️"
        },
        {
            name: "Bajaj Housing Finance",
            type: "NBFC",
            minRate: 8.50, maxRate: 14.00,
            minIncome: 25000, maxLoan: 10000000,
            features: ["Flexi loan", "Part-prepayment free", "Low EMI start"],
            processingFee: "0.40%",
            logo: "📊"
        }
    ];

    function calculateEMI(principal, annualRate, tenureMonths) {
        const r = annualRate / 12 / 100;
        if (r === 0) return principal / tenureMonths;
        return (principal * r * Math.pow(1 + r, tenureMonths)) /
               (Math.pow(1 + r, tenureMonths) - 1);
    }

    function getBankRecommendations(formData, prediction) {
        const loanAmount    = formData.LoanAmount * 1000;   // convert from thousands
        const totalIncome   = formData.ApplicantIncome + formData.CoapplicantIncome;
        const term          = formData.Loan_Amount_Term;
        const creditHistory = formData.Credit_History;
        const probability   = prediction.probability;
        const isApproved    = prediction.status === "APPROVED";

        // Determine applicant's effective rate tier based on credit & probability
        let rateTier = 0.5; // 0 = best rates, 1 = worst rates
        if (creditHistory === 1 && probability > 0.8) {
            rateTier = 0;
        } else if (creditHistory === 1 && probability > 0.6) {
            rateTier = 0.2;
        } else if (creditHistory === 1) {
            rateTier = 0.4;
        } else if (probability > 0.5) {
            rateTier = 0.7;
        } else {
            rateTier = 0.9;
        }

        const offers = INDIAN_BANKS
            .filter(bank => {
                // Filter banks that can serve this loan
                if (loanAmount > bank.maxLoan) return false;
                if (totalIncome < bank.minIncome) return false;
                return true;
            })
            .map(bank => {
                // Calculate effective rate for this applicant
                const effectiveRate = bank.minRate + (bank.maxRate - bank.minRate) * rateTier;
                const emi = calculateEMI(loanAmount, effectiveRate, term);
                const totalPayment = emi * term;
                const totalInterest = totalPayment - loanAmount;

                // Score (lower is better)
                let score = effectiveRate * 10;
                // Bonus for PSU if credit is good
                if (bank.type === "PSU" && creditHistory === 1) score -= 2;
                // Bonus for high max loan capacity
                if (bank.maxLoan >= loanAmount * 2) score -= 1;
                // Penalty for high processing fee
                score += parseFloat(bank.processingFee) * 3;

                return {
                    ...bank,
                    effectiveRate: effectiveRate.toFixed(2),
                    emi: Math.round(emi),
                    totalInterest: Math.round(totalInterest),
                    totalPayment: Math.round(totalPayment),
                    score
                };
            })
            .sort((a, b) => a.score - b.score);

        return offers;
    }

    function renderBankOffers(offers, formData, prediction) {
        const loanAmount = formData.LoanAmount * 1000;
        const term = formData.Loan_Amount_Term;

        // Update eligibility strip
        eligStatus.textContent = prediction.status === "APPROVED" ? "Eligible ✓" : "Not Eligible ✗";
        eligStatus.style.color = prediction.status === "APPROVED" ? "var(--green)" : "var(--red)";
        eligScore.textContent  = prediction.probability_pct;
        eligAmount.textContent = formatINR(loanAmount);
        eligTerm.textContent   = term + " months";

        // Render offers
        bankOffersList.innerHTML = "";

        if (offers.length === 0) {
            bankOffersList.innerHTML = `
                <div class="bank-offer" style="justify-content: center; padding: 2rem;">
                    <p style="color: var(--text-secondary); text-align: center;">
                        No matching bank offers found for your profile.<br/>
                        <span style="font-size: 0.8rem; color: var(--text-muted);">Try adjusting your loan amount or income.</span>
                    </p>
                </div>`;
            return;
        }

        offers.forEach((bank, index) => {
            const rankClass = index === 0 ? "gold" : index === 1 ? "silver" : index === 2 ? "bronze" : "normal";
            const isBest = index === 0;

            const card = document.createElement("div");
            card.className = "bank-offer" + (isBest ? " best" : "");
            card.style.animationDelay = `${index * 0.08}s`;

            card.innerHTML = `
                <div class="bank-offer__rank ${rankClass}">${index + 1}</div>
                <div class="bank-offer__info">
                    <div class="bank-offer__name">
                        <span>${bank.logo} ${bank.name}</span>
                        <span class="bank-offer__type">${bank.type}</span>
                    </div>
                    <div class="bank-offer__details">
                        <span class="bank-offer__detail">Processing: <strong>${bank.processingFee}</strong></span>
                        <span class="bank-offer__detail">Max Loan: <strong>${formatINR(bank.maxLoan)}</strong></span>
                        <span class="bank-offer__detail">Features: <strong>${bank.features[0]}</strong></span>
                    </div>
                </div>
                <div class="bank-offer__rate">
                    <div class="bank-offer__rate-value">${bank.effectiveRate}%</div>
                    <div class="bank-offer__rate-label">p.a.</div>
                    <div class="bank-offer__emi">EMI: ${formatINR(bank.emi)}/mo</div>
                </div>
            `;

            bankOffersList.appendChild(card);
        });
    }

    // ═══════════════════════════════════════════════════════
    // NAVIGATION
    // ═══════════════════════════════════════════════════════

    // View bank offers
    btnBankOffers.addEventListener("click", () => {
        if (!lastPrediction || !lastFormData) return;

        const offers = getBankRecommendations(lastFormData, lastPrediction);
        renderBankOffers(offers, lastFormData, lastPrediction);

        resultCard.classList.add("hidden");
        bankCard.classList.remove("hidden");
    });

    // Back to results
    btnBackResults.addEventListener("click", () => {
        bankCard.classList.add("hidden");
        resultCard.classList.remove("hidden");
    });

    // Reset (from result card)
    btnReset.addEventListener("click", () => {
        resultCard.classList.add("hidden");
        formCard.classList.remove("hidden");
        form.reset();
        lastPrediction = null;
        lastFormData = null;
    });

    // Reset (from bank card)
    btnNewFromBank.addEventListener("click", () => {
        bankCard.classList.add("hidden");
        formCard.classList.remove("hidden");
        form.reset();
        lastPrediction = null;
        lastFormData = null;
    });

    // Forgot password (demo)
    document.getElementById("forgot-password").addEventListener("click", (e) => {
        e.preventDefault();
        loginError.textContent = "Password reset is not available in demo mode.";
        loginError.style.color = "var(--amber)";
        setTimeout(() => {
            loginError.textContent = "";
            loginError.style.color = "";
        }, 3000);
    });

})();
