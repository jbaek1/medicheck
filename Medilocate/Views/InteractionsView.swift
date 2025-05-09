import SwiftUI

struct InteractionsView: View {
    let medication: String
    @State private var interactions: [String] = []
    @State private var isLoading = true
    @State private var errorMessage: String? = nil
    @State private var navigateToContentView = false

    var body: some View {
        NavigationStack {
            VStack {
                if isLoading {
                    ProgressView("Checking interactions...")
                        .padding()
                } else if let errorMessage = errorMessage {
                    Text("Error: \(errorMessage)")
                        .foregroundColor(.red)
                        .padding()
                } else {
                    if interactions.isEmpty {
                        Text("No interactions detected.")
                            .padding()
                        Button("Add Medication", action: addMedicationAndProceed)
                            .padding()
                    } else {
                        Text("Potential Drug Interactions:")
                            .font(.headline)
                            .padding(.top)
                        List(interactions, id: \.self) { interaction in
                            Text(interaction)
                                .padding(.vertical, 4)
                        }
                        Button("Proceed Anyway", action: addMedicationAndProceed)
                            .padding()
                        Button("Cancel", action: proceedToContentView)
                            .padding()
                    }
                }
            }
            .navigationTitle("Interactions for \(medication)")
            .onAppear {
                fetchInteractions()
            }
            .navigationDestination(isPresented: $navigateToContentView) {
                ContentView()
            }
        }
    }

    // Fetch medication interactions
    func fetchInteractions() {
        guard let userId = KeychainHelper.getUserIdentifier() else {
            DispatchQueue.main.async {
                self.errorMessage = "User not found"
                self.isLoading = false
            }
            return
        }

        guard let encodedMedication = medication.addingPercentEncoding(withAllowedCharacters: .urlQueryAllowed),
              let url = URL(string: "\(ContentView.Key.backend_path)interactions?user_id=\(userId)&medication=\(encodedMedication)") else {
            DispatchQueue.main.async {
                self.errorMessage = "Invalid URL"
                self.isLoading = false
            }
            return
        }

        URLSession.shared.dataTask(with: url) { data, response, error in
            if let error = error {
                DispatchQueue.main.async {
                    self.errorMessage = error.localizedDescription
                    self.isLoading = false
                }
                return
            }
            guard let data = data else {
                DispatchQueue.main.async {
                    self.errorMessage = "No data received"
                    self.isLoading = false
                }
                return
            }
            do {
                if let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] {
                    var generatedText: String = ""
                    if let text = json["generated_text"] as? String {
                        generatedText = text
                        print("Generated Text: \(generatedText)")
                    } else {
                        generatedText = String(data: data, encoding: .utf8) ?? ""
                    }
                    let processedLines = processGeneratedText(generatedText)
                    DispatchQueue.main.async {
                        self.interactions = processedLines
                        self.isLoading = false
                    }
                }
            } catch {
                DispatchQueue.main.async {
                    self.errorMessage = error.localizedDescription
                    self.isLoading = false
                }
            }
        }.resume()
    }

    // Process API response
    func processGeneratedText(_ text: String) -> [String] {
        return text.components(separatedBy: "\n")
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
    }

    // Add medication and navigate to ContentView
    func addMedicationAndProceed() {
        addMedication()
        proceedToContentView()
    }

    // Add medication to user's list
    func addMedication() {
        guard let userId = KeychainHelper.getUserIdentifier() else {
            print("User ID not found in Keychain")
            return
        }
        guard let url = URL(string: "\(ContentView.Key.backend_path)users/\(userId)/medications") else {
            print("Invalid URL for updating medications")
            return
        }

        let body: [String: Any] = [
            "medicationsToAdd": [medication],
            "medicationsToRemove": []
        ]

        var request = URLRequest(url: url)
        request.httpMethod = "PATCH"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        do {
            request.httpBody = try JSONSerialization.data(withJSONObject: body)
        } catch {
            print("Error serializing JSON: \(error)")
            return
        }

        URLSession.shared.dataTask(with: request) { data, response, error in
            if let error = error {
                print("Error updating medication: \(error.localizedDescription)")
                return
            }
            guard let data = data else {
                print("No data received when updating medication")
                return
            }
            do {
                if let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] {
                    print("Medication update response: \(json)")
                }
            } catch {
                print("Error parsing medication update response: \(error)")
            }
        }.resume()
    }

    // Navigate to ContentView
    func proceedToContentView() {
        navigateToContentView = true
    }
}

struct InteractionsView_Previews: PreviewProvider {
    static var previews: some View {
        NavigationStack {
            InteractionsView(medication: "TYLENOL")
        }
    }
}
