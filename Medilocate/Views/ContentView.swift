import SwiftUI
import PhotosUI

struct ContentView: View {
    @State private var searchText: String = ""
    @State private var selectedItem: PhotosPickerItem? = nil
    @State private var selectedImage: UIImage?
    @State private var detectedText: String = "No text detected"
    
    // State control vars
    @State private var searchResults: [String]? = nil
    @State private var navigateToPostSearch = false
    
    // Define global vars
    struct Key {
//        static let backend_path = "https://f541-129-59-122-27.ngrok-free.app/api/"
        static let backend_path = "https://medilocatev2.onrender.com/api/"
    }
    
    var body: some View {
        NavigationStack {
            Text("Medication Search 🔍💊")
                .font(.headline)
                .padding(.bottom, 10)
            ZStack {
                Color.blue.opacity(0.2)
                    .ignoresSafeArea()
                
                VStack(spacing: 40) {
                    Text("Which medication are you looking for?")
                        .font(.title2)
                        .multilineTextAlignment(.center)
                    
                    // Search Bar
                    HStack {
                        TextField("Enter medication name...", text: $searchText)
                            .padding()
                            .frame(width: 280, height: 50)
                            .background(Color.white)
                            .cornerRadius(12)
                            .shadow(radius: 2)
                        
                        Button(action: performSearch) {
                            Image(systemName: "magnifyingglass")
                                .foregroundColor(.white)
                                .padding()
                                .background(Color.blue)
                                .clipShape(Circle())
                        }
                        .disabled(searchText.isEmpty)
                    }
                    
                    // Upload Picture
                    PhotosPicker(selection: $selectedItem, matching: .images) {
                        HStack {
                            Image(systemName: "photo.on.rectangle")
                            Text("Upload Medicine Picture")
                        }
                        .foregroundColor(.white)
                        .padding()
                        .background(Color.blue)
                        .cornerRadius(12)
                    }
                    .onChange(of: selectedItem) { _, newItem in
                        Task {
                            if let data = try? await newItem?.loadTransferable(type: Data.self),
                               let uiImage = UIImage(data: data) {
                                selectedImage = uiImage
                                recognizeText(in: uiImage)
                            }
                        }
                    }
                    
                    // Take Picture
                    Button(action: {
                        isCameraPresented.toggle()
                    }) {
                        HStack {
                            Image(systemName: "camera")
                            Text("Take a Picture")
                        }
                        .foregroundColor(.white)
                        .padding()
                        .background(Color.green)
                        .cornerRadius(12)
                    }
                    .sheet(isPresented: $isCameraPresented) {
                        CameraPicker(selectedImage: $selectedImage, onImagePicked: recognizeText)
                    }
                }
                .padding()
                .background(
                    RoundedRectangle(cornerRadius: 20)
                        .fill(Color.white)
                        .shadow(radius: 5)
                )
                .padding(40)
                
                // Navigation to PostSearchView when results are available
                NavigationLink(
                    destination: PostSearchView(searchResults: searchResults ?? []),
                    isActive: $navigateToPostSearch,
                    label: { EmptyView() }
                )
                .hidden()
            }
            BottomNavigationBar()
        }
    }
    

    private let finder = MedicationMatcher()
    // Uses OCR to recognize text, then calls the backend matching function.
    func recognizeText(in image: UIImage) {
        // Immediately dismiss the camera view.
        DispatchQueue.main.async {
            isCameraPresented = false
        }
        
        TextRecognition.shared.recognizeText(in: image) { recognizedText in
            let candidates = recognizedText.components(separatedBy: .whitespacesAndNewlines)
                .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
                .filter { !$0.isEmpty }
            let topCandidates = candidates.prefix(3)
            let query = topCandidates.joined(separator: " ")
            
            detectedText = query
            UserDefaults.standard.set(detectedText, forKey: "ocrResponse")
            
            finder.findClosestMedications(for: detectedText) { results in
                DispatchQueue.main.async {
                    searchResults = results
                    navigateToPostSearch = true
                }
            }
        }
    }
    
    // Medicine Matching for the result in the search bar
    func performSearch() {
        finder.findClosestMedications(for: searchText) { results in
            DispatchQueue.main.async {
                searchResults = results
                navigateToPostSearch = true
            }
        }
    }
    
    @State private var isCameraPresented = false
}

#Preview {
    ContentView()
}
